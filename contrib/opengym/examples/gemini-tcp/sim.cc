#include "tcp-gemini-env.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/opengym-module.h"
#include "ns3/point-to-point-module.h"

using namespace ns3;

int
main(int argc, char* argv[])
{
    uint32_t openGymPort = 5555;
    double simulationTime = 100; // seconds
    Time stepTime = Seconds(1.0);

    CommandLine cmd;
    cmd.AddValue("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
    cmd.AddValue("simTime", "Simulation time in seconds. Default: 100", simulationTime);
    cmd.AddValue("stepTime", "Step time for Gym environment. Default: 1s", stepTime);
    cmd.Parse(argc, argv);

    // Create and configure Gemini environment
    Ptr<GeminiEnv> env = CreateObject<GeminiEnv>(stepTime);
    env->SetOpenGymInterface(CreateObject<OpenGymInterface>(openGymPort));

    // Run simulation
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();

    // Cleanup
    env->NotifySimulationEnd();
    Simulator::Destroy();

    return 0;
}
