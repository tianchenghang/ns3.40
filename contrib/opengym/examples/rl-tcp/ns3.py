def build(bld):
    # Gemini module
    module = bld.create_ns3_module(
        "gemini",
        [
            "network",
            "internet",
            "applications",
            "point-to-point",
            "flow-monitor",
            "opengym",
        ],
    )

    module.source = [
        "model/tcp-gemini.cc",
        "model/gemini-env.cc",
        "helper/gemini-helper.cc",
        "examples/gemini-simulator.cc",
    ]

    module.headers = [
        "model/tcp-gemini.h",
        "model/gemini-env.h",
        "helper/gemini-helper.h",
    ]

    # Add OpenGym dependency if available
    if bld.env["NS3_OPENGYM"]:
        module.use = ["opengym"]
