import Foundation
import Testing
@testable import AutoAnimMacCore

@Suite("Source runtime configuration")
struct RuntimeConfigurationTests {
    @Test("Developer executable accepts an explicit authenticated helper")
    func acceptsDeveloperHelperOverride() throws {
        let configuration = try RuntimeConfiguration.from(
            bundle: .main,
            environment: [
                "AUTOANIM_SOURCE_ROOT": "/tmp/AutoAnim Source",
                "AUTOANIM_NATIVE_HELPER": "/tmp/AutoAnim Source/native/launcher.py",
            ]
        )

        #expect(configuration.sourceRoot.path == "/tmp/AutoAnim Source")
        #expect(configuration.helperScript.path == "/tmp/AutoAnim Source/native/launcher.py")
    }

    @Test("Derives every runtime path from the selected checkout")
    func derivesPaths() throws {
        let root = URL(filePath: "/tmp/AutoAnim Checkout", directoryHint: .isDirectory)
        let helper = URL(filePath: "/Applications/AutoAnim.app/Contents/Resources/source_runtime_service.py")
        let configuration = RuntimeConfiguration(sourceRoot: root, helperScript: helper)

        #expect(configuration.pythonExecutable.path == "/tmp/AutoAnim Checkout/.venv/bin/python")
        #expect(configuration.cliExecutable.path == "/tmp/AutoAnim Checkout/.venv/bin/autoanim-gnm")
        #expect(configuration.artifactRoot.path == "/tmp/AutoAnim Checkout/artifacts/jobs")
        #expect(configuration.viewerVendorDirectory.path.hasSuffix("viewer/three-0.183.2"))
        #expect(configuration.launchArguments.contains("--viewer-vendor"))
        #expect(!configuration.launchArguments.contains("--session-token"))
    }

    @Test("Required and optional dependencies remain distinguishable")
    func validatesRequiredPaths() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString, directoryHint: .isDirectory)
        let helper = root.appending(path: "launcher.py")
        defer { try? FileManager.default.removeItem(at: root) }
        for directory in [
            root.appending(path: ".venv/bin", directoryHint: .isDirectory),
            root.appending(path: ".cache/autoanim_gnm/rhubarb", directoryHint: .isDirectory),
            root.appending(path: ".cache/autoanim_gnm/viewer/three-0.183.2", directoryHint: .isDirectory),
        ] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        for file in [
            root.appending(path: ".venv/bin/python"),
            root.appending(path: ".venv/bin/autoanim-gnm"),
            root.appending(path: ".cache/autoanim_gnm/face_landmarker.task"),
            root.appending(path: ".cache/autoanim_gnm/rhubarb/rhubarb"),
            helper,
        ] {
            try Data("fixture".utf8).write(to: file)
        }
        let configuration = RuntimeConfiguration(sourceRoot: root, helperScript: helper)
        try configuration.validate()
        let diagnostics = configuration.diagnostics()
        #expect(diagnostics.first(where: { $0.label == "A2F runner" })?.required == false)

        try FileManager.default.removeItem(at: helper)
        #expect(throws: RuntimeConfigurationError.self) {
            try configuration.validate()
        }
    }
}
