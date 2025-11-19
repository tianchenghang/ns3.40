#ifndef GEMINI_ENV_H
#define GEMINI_ENV_H

#include "tcp-gemini.h"

#include "ns3/applications-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/node-container.h"
#include "ns3/opengym-module.h"
#include "ns3/point-to-point-channel.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/random-variable-stream.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/traffic-control-helper.h"

#include <map>
#include <vector>

namespace ns3
{

class GeminiEnv : public OpenGymEnv
{
  public:
    GeminiEnv();
    GeminiEnv(Time stepTime);
    virtual ~GeminiEnv();

    static TypeId GetTypeId(void);
    virtual void DoDispose();

    // OpenGymEnv methods
    virtual Ptr<OpenGymSpace> GetActionSpace();
    virtual Ptr<OpenGymSpace> GetObservationSpace();
    virtual bool GetGameOver();
    virtual Ptr<OpenGymDataContainer> GetObservation();
    virtual float GetReward();
    virtual std::string GetExtraInfo();
    virtual bool ExecuteActions(Ptr<OpenGymDataContainer> action);

    // Setup simulation
    void SetupSimulation(uint32_t nNodes = 2);

  private:
    void ResetEnvironment();
    void ScheduleNextStateRead();
    void CollectStatistics();

    NodeContainer m_nodes;
    NetDeviceContainer m_devices;
    Ptr<PointToPointChannel> m_channel;

    Ptr<FlowMonitor> m_flowMonitor;
    FlowMonitorHelper m_flowMonitorHelper;

    Time m_stepTime;
    uint32_t m_stepCount;
    uint32_t m_maxSteps;

    // Gemini parameters
    double m_alpha;
    double m_gamma;
    double m_lambda;
    double m_lossThresh;
    double m_rttThresh;
    uint32_t m_windowSize;

    // Performance metrics
    double m_throughput;
    double m_delay;
    double m_lossRate;
    uint32_t m_flowCount;
};

} // namespace ns3

#endif /* GEMINI_ENV_H */
