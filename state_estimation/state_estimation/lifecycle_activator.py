#!/usr/bin/env python3
"""Bring lifecycle nodes (AMCL) to ACTIVE at start-up -- and never hang doing it.

WHY (jobs 040/041, 2026-10-03)
    nav2's lifecycle manager sends "configure" and then waits for the reply with no
    time limit. In ROS 2 Humble that reply is sometimes lost at start-up ("failed to
    send response to /amcl/change_state (timeout)": AMCL answered before the reply
    channel to the brand-new client was connected). The manager then waited
    forever: AMCL was configured but never activated -> no LiDAR fixes at all, and
    StateEstimate stayed INIT. Starting the manager later did not help.

WHAT THIS NODE DOES INSTEAD: it never relies on a single reply.
    1. ask the node for its state (get_state)
    2. unconfigured -> send configure; inactive -> send activate; active -> done;
       configuring / activating / ... -> wait retry_s and ask again
    3. a request without a reply within call_timeout_s is dropped -> back to 1
    So a lost reply costs call_timeout_s, never forever. Once active, the state is
    read again every watch_s seconds and the node is activated again if it dropped
    out (watch_s 0 = stop checking and exit).
    Unlike nav2's manager there is no bond (heartbeat): if AMCL's process dies,
    the activator only reports that its services are gone.

Parameters: node_names (['amcl']), call_timeout_s (2.0), retry_s (0.5),
            watch_s (5.0), give_up_s (60.0: then ERROR lines, but it keeps trying).

Real car: the same node in the real launch file (AMCL runs on the Pi).
"""
import time

import rclpy
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from rclpy.node import Node

TRANSITION_STATES = (State.TRANSITION_STATE_CONFIGURING, State.TRANSITION_STATE_CLEANINGUP,
                     State.TRANSITION_STATE_SHUTTINGDOWN, State.TRANSITION_STATE_ACTIVATING,
                     State.TRANSITION_STATE_DEACTIVATING, State.TRANSITION_STATE_ERRORPROCESSING)
NAMES = {0: 'unknown', 1: 'unconfigured', 2: 'inactive', 3: 'active', 4: 'finalized',
         10: 'configuring', 11: 'cleaning up', 12: 'shutting down', 13: 'activating',
         14: 'deactivating', 15: 'error processing'}


class Managed:
    """One node being brought up."""

    def __init__(self, node, name):
        self.name = name
        self.get = node.create_client(GetState, f'/{name}/get_state')
        self.change = node.create_client(ChangeState, f'/{name}/change_state')
        self.state = None            # last state read; None = read it again
        self.future = None           # the request in flight
        self.what = ''               # 'get_state', 'configure' or 'activate'
        self.sent = 0.0
        self.next_try = 0.0
        self.lost = 0                # requests that got no reply in time
        self.active = False


class LifecycleActivator(Node):

    def __init__(self):
        super().__init__('lifecycle_activator')
        d = self.declare_parameter
        d('node_names', ['amcl'])
        d('call_timeout_s', 2.0)
        d('retry_s', 0.5)
        d('watch_s', 5.0)
        d('give_up_s', 60.0)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.timeout, self.retry = p('call_timeout_s'), p('retry_s')
        self.watch, self.give_up = p('watch_s'), p('give_up_s')
        self.t0 = time.monotonic()
        self.last_complaint = self.t0
        self.nodes = [Managed(self, n) for n in p('node_names')]
        self.finished = False
        self.create_timer(0.05, self.step)
        self.get_logger().info(
            f"bringing {', '.join(m.name for m in self.nodes)} to ACTIVE (a request without a "
            f"reply within {self.timeout:.2f} s is dropped and the state is read again)")

    def step(self):
        now = time.monotonic()
        for m in self.nodes:
            if m.future is not None:
                if m.future.done():
                    self.on_reply(m, now)
                elif now - m.sent > self.timeout:
                    m.lost += 1
                    self.get_logger().warn(f'{m.name}: no reply to {m.what} within {self.timeout:.2f} s '
                                           f'({m.lost} so far) -> reading its state again')
                    m.future.cancel()
                    m.future, m.state, m.next_try = None, None, now
                continue
            if now < m.next_try:
                continue
            if not (m.get.service_is_ready() and m.change.service_is_ready()):
                self.complain(now, f'{m.name}: lifecycle services not available'
                                   + (' any more' if m.active else ' yet'))
                m.next_try = now + self.retry
                continue
            if m.state is None:
                self.send(m, m.get, GetState.Request(), 'get_state', now)
            elif m.state == State.PRIMARY_STATE_UNCONFIGURED:
                self.send(m, m.change, self.request(Transition.TRANSITION_CONFIGURE), 'configure', now)
            elif m.state == State.PRIMARY_STATE_INACTIVE:
                self.send(m, m.change, self.request(Transition.TRANSITION_ACTIVATE), 'activate', now)
            else:
                m.state, m.next_try = None, now + self.retry
        if not all(m.active for m in self.nodes):
            if now - self.t0 > self.give_up:
                self.complain(now, f'not all nodes ACTIVE after {now - self.t0:.0f} s', error=True)
        elif self.watch <= 0.0:
            self.finished = True

    @staticmethod
    def request(transition_id):
        req = ChangeState.Request()
        req.transition.id = transition_id
        return req

    @staticmethod
    def send(m, client, req, what, now):
        m.future, m.what, m.sent = client.call_async(req), what, now

    def on_reply(self, m, now):
        fut, what = m.future, m.what
        m.future = None
        try:
            res = fut.result()
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'{m.name}: {what} failed ({e}) -> reading its state again')
            m.state, m.next_try = None, now + self.retry
            return
        if what != 'get_state':
            if not res.success:
                self.get_logger().warn(f'{m.name}: {what} refused -> reading its state again')
            m.state, m.next_try = None, now          # always read the state after a transition
            return
        sid = res.current_state.id
        if sid == State.PRIMARY_STATE_ACTIVE:
            if not m.active:
                m.active = True
                extra = f', {m.lost} lost repl{"y" if m.lost == 1 else "ies"} retried' if m.lost else ''
                self.get_logger().info(f'{m.name} ACTIVE after {now - self.t0:.2f} s{extra}')
            m.state, m.next_try = None, now + (self.watch if self.watch > 0.0 else 1e9)
            return
        if m.active:
            self.get_logger().warn(f'{m.name} is {NAMES.get(sid, sid)}, no longer active -> activating again')
            m.active = False
        if sid in TRANSITION_STATES:
            m.state, m.next_try = None, now + self.retry
        elif sid == State.PRIMARY_STATE_FINALIZED:
            self.complain(now, f'{m.name} is finalized (shut down) -- it cannot be activated', error=True)
            m.state, m.next_try = None, now + 5.0
        else:
            m.state, m.next_try = sid, now

    def complain(self, now, text, error=False):
        if now - self.last_complaint < (10.0 if error else 5.0):
            return
        self.last_complaint = now
        (self.get_logger().error if error else self.get_logger().warn)(text)


def main(args=None):
    rclpy.init(args=args)
    node = LifecycleActivator()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
