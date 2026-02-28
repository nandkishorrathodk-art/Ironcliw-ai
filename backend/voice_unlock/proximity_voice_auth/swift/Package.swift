// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "IroncliwProximityAuth",
    platforms: [
        .macOS(.v11)
    ],
    products: [
        .library(
            name: "IroncliwProximityAuth",
            targets: ["IroncliwProximityAuth"]),
        .executable(
            name: "ProximityService",
            targets: ["ProximityService"])
    ],
    dependencies: [
        // No external dependencies for now
    ],
    targets: [
        .target(
            name: "IroncliwProximityAuth",
            dependencies: [],
            path: "Sources/IroncliwProximityAuth"),
        .executableTarget(
            name: "ProximityService",
            dependencies: ["IroncliwProximityAuth"],
            path: "Sources/ProximityService")
    ]
)