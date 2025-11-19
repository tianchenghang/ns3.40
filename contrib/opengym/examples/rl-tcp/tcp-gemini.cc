#include "tcp-gemini.h"

#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/tcp-socket-base.h"

#include <algorithm>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpGemini");
NS_OBJECT_ENSURE_REGISTERED(TcpGemini);

TypeId
TcpGemini::GetTypeId(void)
{
    static TypeId tid = TypeId("ns3::TcpGemini")
                            .SetParent<TcpCongestionOps>()
                            .SetGroupName("Internet")
                            .AddConstructor<TcpGemini>()
                            .AddAttribute("Alpha",
                                          "Multiplicative increase factor",
                                          DoubleValue(1.0),
                                          MakeDoubleAccessor(&TcpGemini::m_alpha),
                                          MakeDoubleChecker<double>())
                            .AddAttribute("Gamma",
                                          "Additive increase factor",
                                          DoubleValue(1.0),
                                          MakeDoubleAccessor(&TcpGemini::m_gamma),
                                          MakeDoubleChecker<double>())
                            .AddAttribute("Lambda",
                                          "Multiplicative decrease factor",
                                          DoubleValue(0.7),
                                          MakeDoubleAccessor(&TcpGemini::m_lambda),
                                          MakeDoubleChecker<double>())
                            .AddAttribute("LossThresh",
                                          "Loss tolerance threshold",
                                          DoubleValue(0.02),
                                          MakeDoubleAccessor(&TcpGemini::m_lossThresh),
                                          MakeDoubleChecker<double>())
                            .AddAttribute("RttThresh",
                                          "RTT inflation threshold",
                                          DoubleValue(0.2),
                                          MakeDoubleAccessor(&TcpGemini::m_rttThresh),
                                          MakeDoubleChecker<double>())
                            .AddAttribute("WindowSize",
                                          "Sliding window size for sampling",
                                          UintegerValue(5),
                                          MakeUintegerAccessor(&TcpGemini::m_windowSize),
                                          MakeUintegerChecker<uint32_t>());
    return tid;
}

TcpGemini::TcpGemini(void)
    : TcpCongestionOps(),
      m_alpha(1.0),
      m_gamma(1.0),
      m_lambda(0.7),
      m_lossThresh(0.02),
      m_rttThresh(0.2),
      m_windowSize(5),
      m_currentThroughput(0.0),
      m_currentLossRate(0.0),
      m_minRtt(Time::Max()),
      m_maxRtt(Time::Min()),
      m_ackedBytes(0),
      m_measurementInterval(Seconds(0.1)),
      m_inSlowStart(true),
      m_inRecovery(false)
{
    NS_LOG_FUNCTION(this);
}

TcpGemini::TcpGemini(const TcpGemini& sock)
    : TcpCongestionOps(sock),
      m_alpha(sock.m_alpha),
      m_gamma(sock.m_gamma),
      m_lambda(sock.m_lambda),
      m_lossThresh(sock.m_lossThresh),
      m_rttThresh(sock.m_rttThresh),
      m_windowSize(sock.m_windowSize),
      m_currentThroughput(sock.m_currentThroughput),
      m_currentLossRate(sock.m_currentLossRate),
      m_minRtt(sock.m_minRtt),
      m_maxRtt(sock.m_maxRtt),
      m_ackedBytes(sock.m_ackedBytes),
      m_measurementInterval(sock.m_measurementInterval),
      m_inSlowStart(sock.m_inSlowStart),
      m_inRecovery(sock.m_inRecovery)
{
    NS_LOG_FUNCTION(this);
}

TcpGemini::~TcpGemini(void)
{
}

std::string
TcpGemini::GetName() const
{
    return "TcpGemini";
}

void
TcpGemini::SetParameters(double alpha,
                         double gamma,
                         double lambda,
                         double lossThresh,
                         double rttThresh,
                         uint32_t windowSize)
{
    m_alpha = alpha;
    m_gamma = gamma;
    m_lambda = lambda;
    m_lossThresh = lossThresh;
    m_rttThresh = rttThresh;
    m_windowSize = windowSize;
}

void
TcpGemini::IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    if (m_inRecovery)
    {
        return;
    }

    if (m_inSlowStart)
    {
        // Slow start: exponential growth
        tcb->m_cWnd += tcb->m_segmentSize * segmentsAcked;
        NS_LOG_INFO("Slow start: cWnd=" << tcb->m_cWnd);
    }
    else
    {
        // Congestion avoidance: mixed increase
        double bdp = m_currentThroughput * m_minRtt.GetSeconds();
        uint32_t targetWindow = std::max(static_cast<uint32_t>(bdp * m_alpha), tcb->m_cWnd.Get());

        if (targetWindow > tcb->m_cWnd)
        {
            // Multiplicative increase
            tcb->m_cWnd = targetWindow;
        }
        else
        {
            // Additive increase
            tcb->m_cWnd += static_cast<uint32_t>(m_gamma * segmentsAcked);
        }
        NS_LOG_INFO("Congestion avoidance: cWnd=" << tcb->m_cWnd);
    }

    // Update statistics
    UpdateStatistics(tcb);
}

uint32_t
TcpGemini::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << tcb << bytesInFlight);

    // Multiplicative decrease
    uint32_t newWindow = static_cast<uint32_t>(tcb->m_cWnd * m_lambda);
    newWindow = std::max(newWindow, 2 * tcb->m_segmentSize);

    // EnterRecovery(const_cast<Ptr<TcpSocketState>>(tcb));
    EnterRecovery(tcb);

    NS_LOG_INFO("Multiplicative decrease: cWnd from " << tcb->m_cWnd << " to " << newWindow);
    return newWindow;
}

void
TcpGemini::PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked << rtt);

    // Update RTT samples
    m_rttSamples.push_back(rtt);
    if (m_rttSamples.size() > m_windowSize)
    {
        m_rttSamples.pop_front();
    }

    // Update min/max RTT
    m_minRtt = Time::Max();
    m_maxRtt = Time::Min();
    for (const auto& sample : m_rttSamples)
    {
        if (sample < m_minRtt)
            m_minRtt = sample;
        if (sample > m_maxRtt)
            m_maxRtt = sample;
    }

    // Check for congestion signals
    if (CheckCongestionSignal())
    {
        EnterRecovery(tcb);
    }

    // Update throughput measurement
    m_ackedBytes += segmentsAcked * tcb->m_segmentSize;
}

void
TcpGemini::CongestionStateSet(Ptr<TcpSocketState> tcb,
                              const TcpSocketState::TcpCongState_t newState)
{
    NS_LOG_FUNCTION(this << tcb << newState);

    if (newState == TcpSocketState::CA_OPEN)
    {
        m_inSlowStart = true;
        m_inRecovery = false;
    }
    else if (newState == TcpSocketState::CA_RECOVERY)
    {
        m_inRecovery = true;
    }
}

Ptr<TcpCongestionOps>
TcpGemini::Fork()
{
    return CopyObject<TcpGemini>(this);
}

void
TcpGemini::UpdateStatistics(Ptr<TcpSocketState> tcb)
{
    // Schedule periodic measurement updates
    if (!m_measurementEvent.IsRunning())
    {
        m_measurementEvent =
            Simulator::Schedule(m_measurementInterval, &TcpGemini::MeasurementTimeout, this);
    }
}

bool
TcpGemini::CheckCongestionSignal(void)
{
    if (m_rttSamples.empty())
    {
        return false;
    }

    // Check RTT inflation
    Time currentRtt = m_rttSamples.back();
    double rttInflation = (currentRtt - m_minRtt).GetSeconds() / (m_maxRtt - m_minRtt).GetSeconds();

    if (rttInflation > m_rttThresh)
    {
        NS_LOG_INFO("Congestion signal: RTT inflation " << rttInflation);
        return true;
    }

    // Check loss rate (simplified)
    if (m_currentLossRate > m_lossThresh)
    {
        NS_LOG_INFO("Congestion signal: Loss rate " << m_currentLossRate);
        return true;
    }

    return false;
}

double
TcpGemini::CalculateCurrentThroughput(void)
{
    if (m_measurementInterval.IsZero())
    {
        return 0.0;
    }

    double throughput = (m_ackedBytes * 8.0) / m_measurementInterval.GetSeconds();
    m_ackedBytes = 0;

    return throughput;
}

void
TcpGemini::MeasurementTimeout(void)
{
    m_currentThroughput = CalculateCurrentThroughput();

    // Update throughput samples
    m_throughputSamples.push_back(m_currentThroughput);
    if (m_throughputSamples.size() > m_windowSize)
    {
        m_throughputSamples.pop_front();
    }

    // Schedule next measurement
    m_measurementEvent =
        Simulator::Schedule(m_measurementInterval, &TcpGemini::MeasurementTimeout, this);
}

void
TcpGemini::EnterSlowStart(Ptr<TcpSocketState> tcb)
{
    m_inSlowStart = true;
    m_inRecovery = false;
    tcb->m_cWnd = tcb->m_segmentSize * 10; // Initial window
}

void
TcpGemini::EnterCongestionAvoidance(Ptr<TcpSocketState> tcb)
{
    m_inSlowStart = false;
    m_inRecovery = false;
}

void
TcpGemini::EnterRecovery(Ptr<TcpSocketState> tcb)
{
    m_inRecovery = true;
    m_inSlowStart = false;
}

void
TcpGemini::GetCurrentStats(double& throughput, double& delay, double& lossRate)
{
    throughput = m_currentThroughput;
    delay = m_minRtt.GetSeconds();
    lossRate = m_currentLossRate;
}

} // namespace ns3
