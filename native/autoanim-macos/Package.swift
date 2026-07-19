// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AutoAnimMac",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "AutoAnimMacCore", targets: ["AutoAnimMacCore"]),
        .executable(name: "AutoAnimMac", targets: ["AutoAnimMac"]),
    ],
    targets: [
        .target(name: "AutoAnimMacCore"),
        .executableTarget(
            name: "AutoAnimMac",
            dependencies: ["AutoAnimMacCore"]
        ),
        .testTarget(
            name: "AutoAnimMacCoreTests",
            dependencies: ["AutoAnimMacCore"]
        ),
    ],
    swiftLanguageModes: [.v5]
)
