#ifndef TCP_GEMINI_H
#define TCP_GEMINI_H

#include "ns3/opengym-module.h"
#include "ns3/random-variable-stream.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-socket-state.h"

#include <deque>
#include <vector>

namespace ns3
{

class TcpGemini : public TcpCongestionOps
{
  public:
    static TypeId GetTypeId(void);

    TcpGemini(void);
    TcpGemini(const TcpGemini& sock);
    virtual ~TcpGemini(void);

    virtual std::string GetName() const;

    virtual void IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);
    virtual uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight);
    virtual void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt);
    virtual void CongestionStateSet(Ptr<TcpSocketState> tcb,
                                    const TcpSocketState::TcpCongState_t newState);

    virtual Ptr<TcpCongestionOps> Fork();

    // Gemini specific methods
    void SetParameters(double alpha,
                       double gamma,
                       double lambda,
                       double lossThresh,
                       double rttThresh,
                       uint32_t windowSize);
    void GetCurrentStats(double& throughput, double& delay, double& lossRate);

  private:
    // Control parameters
    double m_alpha;        // Multiplicative increase factor
    double m_gamma;        // Additive increase factor
    double m_lambda;       // Multiplicative decrease factor
    double m_lossThresh;   // Loss tolerance threshold
    double m_rttThresh;    // RTT inflation threshold
    uint32_t m_windowSize; // Sliding window size for sampling

    // State variables
    std::deque<double> m_throughputSamples;
    std::deque<Time> m_rttSamples;
    std::deque<double> m_lossSamples;

    double m_currentThroughput;
    double m_currentLossRate;
    Time m_minRtt;
    Time m_maxRtt;

    uint32_t m_ackedBytes;
    Time m_measurementInterval;
    EventId m_measurementEvent;

    bool m_inSlowStart;
    bool m_inRecovery;

    // Private methods
    void UpdateStatistics(Ptr<TcpSocketState> tcb);
    bool CheckCongestionSignal(void);
    double CalculateCurrentThroughput(void);
    void MeasurementTimeout(void);
    void EnterSlowStart(Ptr<TcpSocketState> tcb);
    void EnterCongestionAvoidance(Ptr<TcpSocketState> tcb);
    void EnterRecovery(Ptr<TcpSocketState> tcb);
};

} // namespace ns3

#endif /* TCP_GEMINI_H */
