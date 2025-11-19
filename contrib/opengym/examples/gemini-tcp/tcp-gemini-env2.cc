#include "tcp-gemini-env2.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/opengym-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/random-variable-stream.h"

#include <sstream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("GeminiEnv");
NS_OBJECT_ENSURE_REGISTERED(GeminiEnv);

TypeId
GeminiEnv::GetTypeId(void)
{
    static TypeId tid = TypeId("ns3::GeminiEnv")
                            .SetParent<OpenGymEnv>()
                            .SetGroupName("OpenGym")
                            .AddConstructor<GeminiEnv>()
                            .AddAttribute("StepTime",
                                          "The time between each step",
                                          TimeValue(Seconds(1.0)),
                                          MakeTimeAccessor(&GeminiEnv::m_stepTime),
                                          MakeTimeChecker());
    return tid;
}

GeminiEnv::GeminiEnv()
    : m_stepTime(Seconds(1.0)),
      m_stepCount(0),
      m_maxSteps(1000),
      m_alpha(1.0),
      m_gamma(1.0),
      m_lambda(0.7),
      m_lossThresh(0.02),
      m_rttThresh(0.2),
      m_windowSize(5),
      m_throughput(0.0),
      m_delay(0.0),
      m_lossRate(0.0),
      m_flowCount(0)
{
    NS_LOG_FUNCTION(this);
    SetupSimulation();
}

GeminiEnv::GeminiEnv(Time stepTime)
    : m_stepTime(stepTime),
      m_stepCount(0),
      m_maxSteps(1000),
      m_alpha(1.0),
      m_gamma(1.0),
      m_lambda(0.7),
      m_lossThresh(0.02),
      m_rttThresh(0.2),
      m_windowSize(5),
      m_throughput(0.0),
      m_delay(0.0),
      m_lossRate(0.0),
      m_flowCount(0)
{
    NS_LOG_FUNCTION(this);
    SetupSimulation();
}

GeminiEnv::~GeminiEnv()
{
    NS_LOG_FUNCTION(this);
}

void
GeminiEnv::DoDispose()
{
    NS_LOG_FUNCTION(this);
}

void
GeminiEnv::SetupSimulation(uint32_t nNodes)
{
    NS_LOG_FUNCTION(this);

    // Create nodes
    m_nodes.Create(nNodes);

    // Setup point-to-point connection
    PointToPointHelper pointToPoint;
    pointToPoint.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
    pointToPoint.SetChannelAttribute("Delay", StringValue("10ms"));

    m_devices = pointToPoint.Install(m_nodes);
    if (m_devices.GetN() > 0)
    {
        Ptr<NetDevice> dev = m_devices.Get(0);
        if (dev != nullptr)
        {
            Ptr<Channel> ch = dev->GetChannel();
            m_channel = DynamicCast<PointToPointChannel>(ch);
        }
    }

    // Install internet stack
    InternetStackHelper internet;
    internet.Install(m_nodes);

    // Assign IP addresses
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(m_devices);

    // Setup traffic
    uint16_t port = 9;

    // Bulk send application
    BulkSendHelper source("ns3::TcpSocketFactory",
                          InetSocketAddress(interfaces.GetAddress(1), port));
    source.SetAttribute("MaxBytes", UintegerValue(0));
    ApplicationContainer sourceApps = source.Install(m_nodes.Get(0));
    sourceApps.Start(Seconds(1.0));
    sourceApps.Stop(Seconds(100.0));

    // Packet sink application
    PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApps = sink.Install(m_nodes.Get(1));
    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(Seconds(100.0));

    // Setup flow monitor
    m_flowMonitor = m_flowMonitorHelper.InstallAll();

    // Set TCP congestion control to Gemini
    Config::Set("/NodeList/*/$ns3::TcpL4Protocol/SocketType", TypeIdValue(TcpGemini::GetTypeId()));

    // Schedule first step
    Simulator::Schedule(m_stepTime, &GeminiEnv::ScheduleNextStateRead, this);
}

Ptr<OpenGymSpace>
GeminiEnv::GetActionSpace()
{
    NS_LOG_FUNCTION(this);

    // Action space: [alpha, gamma, lambda, lossThresh, rttThresh, windowSize]
    std::vector<uint32_t> shape = {6};
    float low = 0.0;
    float high = 10.0;

    Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low, high, shape);
    return space;
}

Ptr<OpenGymSpace>
GeminiEnv::GetObservationSpace()
{
    NS_LOG_FUNCTION(this);

    // Observation space: [throughput, delay, lossRate, flowCount]
    std::vector<uint32_t> shape = {4};
    float low = 0.0;
    float high = 1000000.0; // 1Gbps max throughput

    Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low, high, shape);
    return space;
}

bool
GeminiEnv::GetGameOver()
{
    NS_LOG_FUNCTION(this);

    bool isGameOver = (m_stepCount >= m_maxSteps);
    if (isGameOver)
    {
        NS_LOG_INFO("Game over at step: " << m_stepCount);
    }

    return isGameOver;
}

Ptr<OpenGymDataContainer>
GeminiEnv::GetObservation()
{
    NS_LOG_FUNCTION(this);

    CollectStatistics();

    std::vector<float> obs = {static_cast<float>(m_throughput),
                              static_cast<float>(m_delay * 1000),   // Convert to ms
                              static_cast<float>(m_lossRate * 100), // Convert to percentage
                              static_cast<float>(m_flowCount)};

    Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>>(obs);
    return box;
}

float
GeminiEnv::GetReward()
{
    NS_LOG_FUNCTION(this);

    // Utility function from Gemini paper: throughput - σ * delay
    double sigma = 0.1;                                      // Preference factor
    double reward = m_throughput - sigma * (m_delay * 1000); // Delay in ms

    NS_LOG_INFO("Reward calculation: throughput=" << m_throughput << " delay=" << m_delay * 1000
                                                  << "ms reward=" << reward);

    return static_cast<float>(reward);
}

std::string
GeminiEnv::GetExtraInfo()
{
    NS_LOG_FUNCTION(this);

    std::stringstream ss;
    ss << "Step: " << m_stepCount << " Throughput: " << m_throughput << " Mbps"
       << " Delay: " << m_delay * 1000 << " ms"
       << " Loss: " << m_lossRate * 100 << "%";

    return ss.str();
}

bool
GeminiEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
    NS_LOG_FUNCTION(this);

    Ptr<OpenGymBoxContainer<float>> box = DynamicCast<OpenGymBoxContainer<float>>(action);
    if (box == nullptr)
    {
        NS_LOG_ERROR("Invalid action type");
        return false;
    }

    std::vector<float> actions = box->GetData();

    // Update Gemini parameters (with normalization)
    m_alpha = std::max(1.0, static_cast<double>(actions[0]));                        // >= 1.0
    m_gamma = std::max(1.0, static_cast<double>(actions[1]));                        // >= 1.0
    m_lambda = std::min(0.9, std::max(0.5, static_cast<double>(actions[2]) / 10.0)); // 0.5-0.9
    m_lossThresh =
        std::min(0.1, std::max(0.001, static_cast<double>(actions[3]) / 100.0));        // 0.1%-10%
    m_rttThresh = std::min(0.5, std::max(0.1, static_cast<double>(actions[4]) / 20.0)); // 0.1-0.5
    m_windowSize = std::min(10u, std::max(1u, static_cast<uint32_t>(actions[5])));      // 1-10

    NS_LOG_INFO("New parameters - alpha: " << m_alpha << " gamma: " << m_gamma << " lambda: "
                                           << m_lambda << " lossThresh: " << m_lossThresh
                                           << " rttThresh: " << m_rttThresh
                                           << " windowSize: " << m_windowSize);

    // Apply parameters to all TCP sockets
    for (uint32_t i = 0; i < m_nodes.GetN(); ++i)
    {
        Ptr<Node> node = m_nodes.Get(i);
        Ptr<TcpL4Protocol> tcp = node->GetObject<TcpL4Protocol>();

        if (tcp)
        {
            ObjectVectorValue sockets;
            tcp->GetAttribute("SocketList", sockets);

            for (auto it = sockets.Begin(); it != sockets.End(); ++it)
            {
                Ptr<TcpSocketBase> socket = DynamicCast<TcpSocketBase>(it->second);
                if (socket)
                {
                    Ptr<TcpGemini> gemini = DynamicCast<TcpGemini>(socket);
                    if (gemini)
                    {
                        gemini->SetParameters(m_alpha,
                                              m_gamma,
                                              m_lambda,
                                              m_lossThresh,
                                              m_rttThresh,
                                              m_windowSize);
                    }
                }
            }
        }
    }

    m_stepCount++;
    return true;
}

void
GeminiEnv::ScheduleNextStateRead()
{
    NS_LOG_FUNCTION(this);

    if (!GetGameOver())
    {
        Notify();
        Simulator::Schedule(m_stepTime, &GeminiEnv::ScheduleNextStateRead, this);
    }
}

void
GeminiEnv::CollectStatistics()
{
    NS_LOG_FUNCTION(this);

    m_flowMonitor->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer stats = m_flowMonitor->GetFlowStats();

    double totalThroughput = 0.0;
    double totalDelay = 0.0;
    double totalLossRate = 0.0;
    uint32_t flowCount = 0;

    for (auto it = stats.begin(); it != stats.end(); ++it)
    {
        //// FlowId flowId = it->first;
        FlowMonitor::FlowStats flowStats = it->second;

        if (flowStats.rxPackets > 0)
        {
            // Calculate throughput in Mbps
            double throughput =
                (flowStats.rxBytes * 8.0) /
                (flowStats.timeLastRxPacket - flowStats.timeFirstTxPacket).GetSeconds() / 1e6;

            // Calculate average delay in seconds
            double delay = flowStats.delaySum.GetSeconds() / flowStats.rxPackets;

            // Calculate loss rate
            double lossRate = static_cast<double>(flowStats.lostPackets) /
                              (flowStats.rxPackets + flowStats.lostPackets);

            totalThroughput += throughput;
            totalDelay += delay;
            totalLossRate += lossRate;
            flowCount++;
        }
    }

    if (flowCount > 0)
    {
        m_throughput = totalThroughput;
        m_delay = totalDelay / flowCount;
        m_lossRate = totalLossRate / flowCount;
        m_flowCount = flowCount;
    }
    else
    {
        m_throughput = 0.0;
        m_delay = 0.0;
        m_lossRate = 0.0;
        m_flowCount = 0;
    }

    NS_LOG_DEBUG("Statistics - Flows: " << flowCount << " Throughput: " << m_throughput << " Mbps"
                                        << " Delay: " << m_delay * 1000 << " ms"
                                        << " Loss: " << m_lossRate * 100 << "%");
}

} // namespace ns3
