__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2018, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "gawlowicz@tkn.tu-berlin.de"


class Tcp(object):
    """docstring for Tcp"""

    def __init__(self):
        super(Tcp, self).__init__()

    def set_spaces(self, obs, act):
        self.obsSpace = obs
        self.actSpace = act

    def get_action(self, obs, reward, done, info):
        pass


class TcpEventBased(Tcp):
    """docstring for TcpEventBased"""

    def __init__(self):
        super(TcpEventBased, self).__init__()

    def get_action(self, obs, reward, done, info):
        # unique socket ID
        socketUuid = obs[0]
        # TCP env type: event-based = 0 / time-based = 1
        envType = obs[1]
        # sim time in us
        simTime_us = obs[2]
        # unique node ID
        nodeId = obs[3]
        # current ssThreshold
        ssThresh = obs[4]
        # current contention window size
        cWnd = obs[5]
        # segment size
        segmentSize = obs[6]
        # number of acked segments
        segmentsAcked = obs[7]
        # estimated bytes in flight
        bytesInFlight = obs[8]
        # last estimation of RTT
        lastRtt_us = obs[9]
        # min value of RTT
        minRtt_us = obs[10]
        # function from Congestion Algorithm (CA) interface:
        #  GET_SS_THRESH = 0 (packet loss),
        #  INCREASE_WINDOW (packet acked),
        #  PKTS_ACKED (unused),
        #  CONGESTION_STATE_SET (unused),
        #  CWND_EVENT (unused),
        calledFunc = obs[11]
        # Congetsion Algorithm (CA) state:
        #  CA_OPEN = 0,
        #  CA_DISORDER,
        #  CA_CWR,
        #  CA_RECOVERY,
        #  CA_LOSS,
        #  CA_LAST_STATE
        caState = obs[12]
        # Congetsion Algorithm (CA) event:
        #  CA_EVENT_TX_START = 0,
        #  CA_EVENT_CWND_RESTART,
        #  CA_EVENT_COMPLETE_CWR,
        #  CA_EVENT_LOSS,
        #  CA_EVENT_ECN_NO_CE,
        #  CA_EVENT_ECN_IS_CE,
        #  CA_EVENT_DELAYED_ACK,
        #  CA_EVENT_NON_DELAYED_ACK,
        caEvent = obs[13]
        # ECN state:
        #  ECN_DISABLED = 0,
        #  ECN_IDLE,
        #  ECN_CE_RCVD,
        #  ECN_SENDING_ECE,
        #  ECN_ECE_RCVD,
        #  ECN_CWR_SENT
        ecnState = obs[14]

        # 拥塞事件常量
        CA_EVENT_LOSS = 3
        CA_EVENT_ECN_IS_CE = 5
        # ECN状态常量
        ECN_CE_RCVD = 2
        ECN_SENDING_ECE = 3

        min_cwnd = 2 * segmentSize
        max_cwnd = 100 * segmentSize

        # 先根据caEvent和ecnState判断是否需要强制收敛
        if caEvent == CA_EVENT_LOSS:
            # 丢包事件,大幅减小窗口
            new_cWnd = max(min_cwnd, cWnd * 0.5)
        elif caEvent == CA_EVENT_ECN_IS_CE or ecnState in (
            ECN_CE_RCVD,
            ECN_SENDING_ECE,
        ):
            # ECN检测到拥塞,适度减小
            new_cWnd = max(min_cwnd, cWnd * 0.7)
        elif caState == 0:  # CA_OPEN
            if lastRtt_us > 2 * minRtt_us:
                # RTT大幅增加,减小窗口
                new_cWnd = max(min_cwnd, cWnd * 0.85)
            elif bytesInFlight > cWnd * 0.95:
                # 发送窗口接近满,保持不变
                new_cWnd = int(cWnd)
            else:
                # 网络良好,适度增加
                new_cWnd = min(max_cwnd, cWnd * 1.08)
        else:
            # 其他拥塞状态,适度减小
            new_cWnd = max(min_cwnd, cWnd * 0.7)

        # ssThresh动态调整
        if caEvent == CA_EVENT_LOSS or caState != 0:
            new_ssThresh = max(min_cwnd, int(bytesInFlight * 0.7))
        else:
            new_ssThresh = max(min_cwnd, int(ssThresh * 0.95))

        # 限制窗口范围
        if new_cWnd > max_cwnd:
            new_cWnd = max_cwnd
        if new_cWnd < min_cwnd:
            new_cWnd = min_cwnd

        return [new_ssThresh, new_cWnd]


class TcpTimeBased(Tcp):
    """docstring for TcpTimeBased"""

    def __init__(self):
        super(TcpTimeBased, self).__init__()

    def get_action(self, obs, reward, done, info):
        # unique socket ID
        socketUuid = obs[0]
        # TCP env type: event-based = 0 / time-based = 1
        envType = obs[1]
        # sim time in us
        simTime_us = obs[2]
        # unique node ID
        nodeId = obs[3]
        # current ssThreshold
        ssThresh = obs[4]
        # current contention window size
        cWnd = obs[5]
        # segment size
        segmentSize = obs[6]
        # bytesInFlightSum
        bytesInFlightSum = obs[7]
        # bytesInFlightAvg
        bytesInFlightAvg = obs[8]
        # segmentsAckedSum
        segmentsAckedSum = obs[9]
        # segmentsAckedAvg
        segmentsAckedAvg = obs[10]
        # avgRtt
        avgRtt = obs[11]
        # minRtt
        minRtt = obs[12]
        # avgInterTx
        avgInterTx = obs[13]
        # avgInterRx
        avgInterRx = obs[14]
        # throughput
        throughput = obs[15]

        # 优化: 根据RTT、吞吐量、bytesInFlight等动态调整
        min_cwnd = 2 * segmentSize
        max_cwnd = 100 * segmentSize

        # 以吞吐量和RTT为主要指标
        if avgRtt > 0 and avgRtt > 2 * minRtt:
            # RTT大幅增加,减小窗口
            new_cWnd = max(min_cwnd, cWnd * 0.8)
        elif throughput < 0.8 * (cWnd / avgRtt) * segmentSize and avgRtt > 0:
            # 吞吐量低于理论,减小窗口
            new_cWnd = max(min_cwnd, cWnd * 0.9)
        elif bytesInFlightAvg > 0.95 * cWnd:
            # 发送窗口接近满,保持不变
            new_cWnd = int(cWnd)
        else:
            # 网络良好,适度增加
            new_cWnd = min(max_cwnd, cWnd * 1.05)

        # ssThresh动态调整
        if avgRtt > 0 and avgRtt > 2 * minRtt:
            new_ssThresh = max(min_cwnd, int(bytesInFlightAvg * 0.7))
        else:
            new_ssThresh = max(min_cwnd, int(ssThresh * 0.95))

        # 限制窗口范围
        if new_cWnd > max_cwnd:
            new_cWnd = max_cwnd
        if new_cWnd < min_cwnd:
            new_cWnd = min_cwnd

        return [new_ssThresh, new_cWnd]
