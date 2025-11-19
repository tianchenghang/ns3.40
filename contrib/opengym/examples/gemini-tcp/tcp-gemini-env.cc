#include "tcp-gemini-env.h"

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
    static TypeId tid =
        TypeId("ns3::GeminiEnv")
            .SetParent<OpenGymEnv>()
            .SetGroupName("OpenGym")
            .AddConstructor<GeminiEnv>()
            .AddAttribute("StepTime",
                          "The time between each step",
                          TimeValue(Seconds(1.0)),
                          MakeTimeAccessor(&GeminiEnv::m_stepTime),
                          MakeTimeChecker())
            .AddAttribute(
                "NetworkType",
                "Network environment type (0=Ethernet, 1=Cellular, 2=Satellite, 3=Random)",
                UintegerValue(0),
                MakeUintegerAccessor(&GeminiEnv::m_networkType),
                MakeUintegerChecker<uint32_t>())
            .AddAttribute("MaxSteps",
                          "Maximum number of steps per episode",
                          UintegerValue(500),
                          MakeUintegerAccessor(&GeminiEnv::m_maxSteps),
                          MakeUintegerChecker<uint32_t>());
    return tid;
}

GeminiEnv::GeminiEnv()
    : m_stepTime(Seconds(1.0)),
      m_stepCount(0),
      m_maxSteps(500),
      m_networkType(0),
      m_useRandomEnv(false),
      m_alpha(1.0),
      m_gamma(1.0),
      m_lambda(0.7),
      m_lossThresh(0.02),
      m_rttThresh(0.2),
      m_windowSize(5),
      m_throughput(0.0),
      m_delay(0.0),
      m_lossRate(0.0),
      m_flowCount(0),
      m_utility(0.0),
      m_currentFlowId(0)
{
    NS_LOG_FUNCTION(this);

    m_uniformRandom = CreateObject<UniformRandomVariable>();

    // Set hidden variables (simplified)
    m_region = "east";
    m_isp = "telecom";
    m_timeOfDay = 12; // noon

    // Setup specific network environment
    switch (m_networkType)
    {
    case 0:
        SetupEthernetEnvironment();
        break;
    case 1:
        SetupCellularEnvironment();
        break;
    case 2:
        SetupSatelliteEnvironment();
        break;
    case 3:
        SetupRandomEnvironment();
        break;
    default:
        SetupEthernetEnvironment();
    }
}

GeminiEnv::GeminiEnv(Time stepTime, uint32_t networkType)
    : m_stepTime(stepTime),
      m_stepCount(0),
      m_maxSteps(500),
      m_networkType(networkType),
      m_useRandomEnv(networkType == 3),
      m_alpha(1.0),
      m_gamma(1.0),
      m_lambda(0.7),
      m_lossThresh(0.02),
      m_rttThresh(0.2),
      m_windowSize(5),
      m_throughput(0.0),
      m_delay(0.0),
      m_lossRate(0.0),
      m_flowCount(0),
      m_utility(0.0),
      m_currentFlowId(0)
{
    NS_LOG_FUNCTION(this);

    m_uniformRandom = CreateObject<UniformRandomVariable>();

    // Set hidden variables
    m_region = "east";
    m_isp = "telecom";
    m_timeOfDay = 12;

    switch (m_networkType)
    {
    case 0:
        SetupEthernetEnvironment();
        break;
    case 1:
        SetupCellularEnvironment();
        break;
    case 2:
        SetupSatelliteEnvironment();
        break;
    case 3:
        SetupRandomEnvironment();
        break;
    default:
        SetupEthernetEnvironment();
    }
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
GeminiEnv::SetupEthernetEnvironment()
{
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO("Setting up Ethernet environment");

    SetupNetwork("15Mbps", "150ms", 100, 0.0);
    GenerateTraffic();
}

void
GeminiEnv::SetupCellularEnvironment()
{
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO("Setting up Cellular environment");

    // Cellular networks have variable RTT and bandwidth
    SetupNetwork("15Mbps", "100ms", 50, 0.05); // 5% loss rate

    // Schedule dynamic network changes for cellular
    for (uint32_t i = 0; i < 10; ++i)
    {
        Time changeTime = Seconds(m_uniformRandom->GetInteger(5, 50));
        Simulator::Schedule(changeTime, &GeminiEnv::UpdateNetworkConditions, this);
    }

    GenerateTraffic();
}

void
GeminiEnv::SetupSatelliteEnvironment()
{
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO("Setting up Satellite environment");

    SetupNetwork("45Mbps", "800ms", 500, 0.0074); // 0.74% loss rate
    GenerateTraffic();
}

void
GeminiEnv::SetupRandomEnvironment()
{
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO("Setting up Random environment");

    // Random parameters within typical ranges
    std::vector<std::string> dataRates = {"1Mbps", "10Mbps", "50Mbps", "100Mbps"};
    std::vector<std::string> delays = {"10ms", "50ms", "100ms", "200ms", "500ms"};
    std::vector<uint32_t> queueSizes = {10, 50, 100, 200, 500};

    uint32_t drIndex = m_uniformRandom->GetInteger(0, dataRates.size() - 1);
    uint32_t dIndex = m_uniformRandom->GetInteger(0, delays.size() - 1);
    uint32_t qIndex = m_uniformRandom->GetInteger(0, queueSizes.size() - 1);
    double lossRate = m_uniformRandom->GetValue(0.0, 0.1); // 0-10% loss

    SetupNetwork(dataRates[drIndex], delays[dIndex], queueSizes[qIndex], lossRate);
    GenerateTraffic();
}

void
GeminiEnv::SetupNetwork(const std::string& dataRate,
                        const std::string& delay,
                        uint32_t queueSize,
                        double lossRate)
{
    NS_LOG_FUNCTION(this << dataRate << delay << queueSize << lossRate);

    // Create nodes
    m_nodes.Create(2);

    // Setup point-to-point connection
    PointToPointHelper pointToPoint;
    pointToPoint.SetDeviceAttribute("DataRate", StringValue(dataRate));
    pointToPoint.SetChannelAttribute("Delay", StringValue(delay));

    // Setup queue discipline
    m_tch.SetRootQueueDisc("ns3::PfifoFastQueueDisc",
                           "MaxSize",
                           StringValue(std::to_string(queueSize) + "p"));

    m_devices = pointToPoint.Install(m_nodes);

    // Add packet loss if specified
    if (lossRate > 0.0)
    {
        Ptr<RateErrorModel> em = CreateObject<RateErrorModel>();
        em->SetAttribute("ErrorRate", DoubleValue(lossRate));
        m_devices.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(em));
    }

    // Install internet stack
    InternetStackHelper internet;
    internet.Install(m_nodes);

    // Assign IP addresses
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(m_devices);

    // Install traffic control
    m_tch.Install(m_devices);

    // Setup flow monitor
    m_flowMonitor = m_flowMonitorHelper.InstallAll();

    // Set TCP congestion control to Gemini
    Config::Set("/NodeList/*/$ns3::TcpL4Protocol/SocketType", TypeIdValue(TcpGemini::GetTypeId()));

    NS_LOG_INFO("Network setup complete: " << dataRate << ", " << delay << ", queue=" << queueSize
                                           << "p, loss=" << lossRate);
}

void
GeminiEnv::GenerateTraffic()
{
    NS_LOG_FUNCTION(this);

    uint16_t port = 9;
    // 获取节点 1 接口 1 的本地 IPv4 地址
    Ipv4Address remoteAddr = m_nodes.Get(1)->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();

    // Create multiple flows with different characteristics
    for (uint32_t i = 0; i < 5; ++i)
    {
        Time startTime = Seconds(1.0 + i * 0.5);

        // Random flow size: 100KB to 10MB
        uint32_t maxBytes = m_uniformRandom->GetInteger(100000, 10000000);

        BulkSendHelper source("ns3::TcpSocketFactory", InetSocketAddress(remoteAddr, port + i));
        source.SetAttribute("MaxBytes", UintegerValue(maxBytes));

        ApplicationContainer sourceApp = source.Install(m_nodes.Get(0));
        sourceApp.Start(startTime);
        sourceApp.Stop(Seconds(100.0));

        // Packet sink
        PacketSinkHelper sink("ns3::TcpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port + i));
        ApplicationContainer sinkApp = sink.Install(m_nodes.Get(1));
        sinkApp.Start(Seconds(0.0));
        sinkApp.Stop(Seconds(100.0));

        m_flowStartTimes[m_currentFlowId++] = startTime;
    }

    // Schedule first step
    Simulator::Schedule(m_stepTime, &GeminiEnv::ScheduleNextStateRead, this);
}

void
GeminiEnv::UpdateNetworkConditions()
{
    NS_LOG_FUNCTION(this);

    if (m_useRandomEnv)
    {
        // Change network conditions dynamically (for cellular/random environments)
        Ptr<PointToPointNetDevice> device = DynamicCast<PointToPointNetDevice>(m_devices.Get(0));
        if (device)
        {
            // Randomly change data rate
            std::vector<std::string> dataRates = {"1Mbps", "5Mbps", "10Mbps", "15Mbps", "20Mbps"};
            uint32_t index = m_uniformRandom->GetInteger(0, dataRates.size() - 1);
            device->SetDataRate(DataRate(dataRates[index]));

            NS_LOG_INFO("Changed data rate to: " << dataRates[index]);
        }

        // Schedule next change
        Time nextChange = Seconds(m_uniformRandom->GetInteger(5, 20));
        Simulator::Schedule(nextChange, &GeminiEnv::UpdateNetworkConditions, this);
    }
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

    // Observation space: [throughput, delay, lossRate, flowCount, timeOfDay]
    std::vector<uint32_t> shape = {5};
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

    // Normalize observations for better learning
    double normThroughput = m_throughput / 100.0; // Normalize to 0-10
    double normDelay = m_delay * 1000.0 / 100.0;  // Normalize to 0-10 (100ms max)
    double normLossRate = m_lossRate * 100.0;     // Percentage
    double normFlowCount = m_flowCount / 10.0;    // Normalize to 0-10
    double normTimeOfDay = m_timeOfDay / 24.0;    // Normalize to 0-1

    std::vector<float> obs = {static_cast<float>(normThroughput),
                              static_cast<float>(normDelay),
                              static_cast<float>(normLossRate),
                              static_cast<float>(normFlowCount),
                              static_cast<float>(normTimeOfDay)};

    Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>>(obs);
    return box;
}

float
GeminiEnv::GetReward()
{
    NS_LOG_FUNCTION(this);

    // Gemini utility function: throughput - σ * delay
    double sigma = 0.1;                                    // Preference factor from paper
    m_utility = m_throughput - sigma * (m_delay * 1000.0); // Delay in ms

    // Add penalty for high loss rate
    if (m_lossRate > 0.1)
    { // 10% loss threshold
        m_utility *= (1.0 - m_lossRate);
    }

    NS_LOG_DEBUG("Reward calculation: throughput=" << m_throughput << " delay=" << m_delay * 1000
                                                   << "ms utility=" << m_utility);

    return static_cast<float>(m_utility);
}

std::string
GeminiEnv::GetExtraInfo()
{
    NS_LOG_FUNCTION(this);

    std::stringstream ss;
    ss << "Step: " << m_stepCount << " Network: " << m_networkType
       << " Throughput: " << m_throughput << " Mbps"
       << " Delay: " << m_delay * 1000 << " ms"
       << " Loss: " << m_lossRate * 100 << "%"
       << " Flows: " << m_flowCount << " Utility: " << m_utility;

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

    // Update Gemini parameters with proper normalization
    m_alpha = 1.0 + (actions[0] / 5.0);                         // 1.0 - 3.0
    m_gamma = 1.0 + (actions[1] / 2.0);                         // 1.0 - 6.0
    m_lambda = 0.5 + (actions[2] / 20.0);                       // 0.5 - 1.0
    m_lossThresh = 0.001 + (actions[3] / 1000.0);               // 0.001 - 0.011
    m_rttThresh = 0.1 + (actions[4] / 20.0);                    // 0.1 - 0.6
    m_windowSize = 1 + static_cast<uint32_t>(actions[5] / 2.0); // 1 - 6

    // Apply constraints
    m_alpha = std::min(3.0, std::max(1.0, m_alpha));
    m_gamma = std::min(6.0, std::max(1.0, m_gamma));
    m_lambda = std::min(0.95, std::max(0.5, m_lambda));
    m_lossThresh = std::min(0.05, std::max(0.001, m_lossThresh));
    m_rttThresh = std::min(0.8, std::max(0.1, m_rttThresh));
    m_windowSize = std::min(10u, std::max(1u, m_windowSize));

    NS_LOG_INFO("New Gemini parameters - alpha: " << m_alpha << " gamma: " << m_gamma << " lambda: "
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

    // Update time of day (for hidden variables)
    m_timeOfDay = (m_timeOfDay + 1) % 24;

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
    uint32_t validFlows = 0;

    Time currentTime = Simulator::Now();

    for (auto it = stats.begin(); it != stats.end(); ++it)
    {
        FlowMonitor::FlowStats flowStats = it->second;

        // Only consider flows that have completed or are in progress
        if (flowStats.rxPackets > 0 || flowStats.txPackets > 0)
        {
            double throughput = 0.0;
            double delay = 0.0;
            double lossRate = 0.0;

            if (flowStats.rxPackets > 0)
            {
                // Calculate throughput in Mbps
                throughput = (flowStats.rxBytes * 8.0) /
                             (currentTime - flowStats.timeFirstTxPacket).GetSeconds() / 1e6;

                // Calculate average delay in seconds
                delay = flowStats.delaySum.GetSeconds() / flowStats.rxPackets;

                // Calculate loss rate
                lossRate = static_cast<double>(flowStats.lostPackets) /
                           (flowStats.rxPackets + flowStats.lostPackets);

                totalThroughput += throughput;
                totalDelay += delay;
                totalLossRate += lossRate;
                validFlows++;
            }
        }
    }

    if (validFlows > 0)
    {
        m_throughput = totalThroughput;
        m_delay = totalDelay / validFlows;
        m_lossRate = totalLossRate / validFlows;
        m_flowCount = validFlows;
    }
    else
    {
        m_throughput = 0.0;
        m_delay = 0.0;
        m_lossRate = 0.0;
        m_flowCount = 0;
    }

    // Update history
    m_throughputHistory.push_back(m_throughput);
    m_delayHistory.push_back(m_delay);
    m_lossRateHistory.push_back(m_lossRate);

    // Keep history size manageable
    if (m_throughputHistory.size() > 100)
    {
        m_throughputHistory.erase(m_throughputHistory.begin());
        m_delayHistory.erase(m_delayHistory.begin());
        m_lossRateHistory.erase(m_lossRateHistory.begin());
    }

    NS_LOG_DEBUG("Statistics - Flows: " << validFlows << " Throughput: " << m_throughput << " Mbps"
                                        << " Delay: " << m_delay * 1000 << " ms"
                                        << " Loss: " << m_lossRate * 100 << "%");
}

void
GeminiEnv::SetGeminiParameters(double alpha,
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
GeminiEnv::GetCurrentGeminiParameters(double& alpha,
                                      double& gamma,
                                      double& lambda,
                                      double& lossThresh,
                                      double& rttThresh,
                                      uint32_t& windowSize)
{
    alpha = m_alpha;
    gamma = m_gamma;
    lambda = m_lambda;
    lossThresh = m_lossThresh;
    rttThresh = m_rttThresh;
    windowSize = m_windowSize;
}

} // namespace ns3
