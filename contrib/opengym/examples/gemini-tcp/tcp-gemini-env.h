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
    GeminiEnv(Time stepTime, uint32_t networkType = 0);
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

    // Network configuration
    void SetupEthernetEnvironment();
    void SetupCellularEnvironment();
    void SetupSatelliteEnvironment();
    void SetupRandomEnvironment();

    // Parameter management
    void SetGeminiParameters(double alpha,
                             double gamma,
                             double lambda,
                             double lossThresh,
                             double rttThresh,
                             uint32_t windowSize);
    void GetCurrentGeminiParameters(double& alpha,
                                    double& gamma,
                                    double& lambda,
                                    double& lossThresh,
                                    double& rttThresh,
                                    uint32_t& windowSize);

  private:
    void ResetEnvironment();
    void ScheduleNextStateRead();
    void CollectStatistics();
    void SetupNetwork(const std::string& dataRate,
                      const std::string& delay,
                      uint32_t queueSize,
                      double lossRate = 0.0);
    void GenerateTraffic();
    void UpdateNetworkConditions();

    // Network components
    NodeContainer m_nodes;
    NetDeviceContainer m_devices;
    Ptr<PointToPointChannel> m_channel;
    TrafficControlHelper m_tch;

    Ptr<FlowMonitor> m_flowMonitor;
    FlowMonitorHelper m_flowMonitorHelper;

    // Simulation parameters
    Time m_stepTime;
    uint32_t m_stepCount;
    uint32_t m_maxSteps;
    uint32_t m_networkType;
    bool m_useRandomEnv;

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
    double m_utility;

    // Traffic generation
    Ptr<UniformRandomVariable> m_uniformRandom;
    uint32_t m_currentFlowId;
    std::map<uint32_t, Time> m_flowStartTimes;

    // Hidden variables (for Booster)
    std::string m_region;
    std::string m_isp;
    uint32_t m_timeOfDay;

    // Statistics history
    std::vector<double> m_throughputHistory;
    std::vector<double> m_delayHistory;
    std::vector<double> m_lossRateHistory;
};

} // namespace ns3

#endif /* GEMINI_ENV_H */
