// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "A2FRunner",
    platforms: [
        .macOS("15.0")
    ],
    products: [
        .executable(name: "a2f-runner", targets: ["A2FRunner"])
    ],
    dependencies: [
        .package(
            url: "https://github.com/soniqo/speech-swift.git",
            exact: "0.0.23"
        )
    ],
    targets: [
        .target(name: "A2FRunnerCore"),
        .executableTarget(
            name: "A2FRunner",
            dependencies: [
                "A2FRunnerCore",
                .product(name: "Audio2Face3D", package: "speech-swift"),
                .product(name: "AudioCommon", package: "speech-swift")
            ]
        ),
        .testTarget(
            name: "A2FRunnerCoreTests",
            dependencies: ["A2FRunnerCore"]
        )
    ]
)
