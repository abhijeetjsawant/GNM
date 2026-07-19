import Foundation

public struct RuntimeConfiguration: Equatable, Sendable {
    public let sourceRoot: URL
    public let pythonExecutable: URL
    public let cliExecutable: URL
    public let helperScript: URL
    public let artifactRoot: URL
    public let modelPath: URL
    public let rhubarbExecutable: URL
    public let a2fRunner: URL
    public let a2fAssetDirectory: URL
    public let viewerVendorDirectory: URL

    public init(sourceRoot: URL, helperScript: URL) {
        let root = sourceRoot.standardizedFileURL
        self.sourceRoot = root
        self.pythonExecutable = root.appending(path: ".venv/bin/python")
        self.cliExecutable = root.appending(path: ".venv/bin/autoanim-gnm")
        self.helperScript = helperScript.standardizedFileURL
        self.artifactRoot = root.appending(path: "artifacts/jobs")
        self.modelPath = root.appending(path: ".cache/autoanim_gnm/face_landmarker.task")
        self.rhubarbExecutable = root.appending(path: ".cache/autoanim_gnm/rhubarb/rhubarb")
        self.a2fRunner = root.appending(path: "native/a2f-runner/.build/arm64-apple-macosx/release/a2f-runner")
        self.a2fAssetDirectory = root.appending(path: ".cache/autoanim_gnm/a2f-claire")
        self.viewerVendorDirectory = root.appending(path: ".cache/autoanim_gnm/viewer/three-0.183.2")
    }

    public static func from(
        bundle: Bundle = .main,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeConfiguration {
        let suppliedRoot = environment["AUTOANIM_SOURCE_ROOT"]
            ?? bundle.object(forInfoDictionaryKey: "AutoAnimSourceRoot") as? String
        guard let suppliedRoot, !suppliedRoot.isEmpty, suppliedRoot != "__SOURCE_ROOT__" else {
            throw RuntimeConfigurationError.missingSourceRoot
        }
        let helper: URL
        if let suppliedHelper = environment["AUTOANIM_NATIVE_HELPER"], !suppliedHelper.isEmpty {
            helper = URL(filePath: suppliedHelper)
        } else if let bundledHelper = bundle.resourceURL?.appending(path: "source_runtime_service.py") {
            helper = bundledHelper
        } else {
            throw RuntimeConfigurationError.missingBundleResources
        }
        return RuntimeConfiguration(
            sourceRoot: URL(filePath: suppliedRoot, directoryHint: .isDirectory),
            helperScript: helper
        )
    }

    public var launchArguments: [String] {
        [
            helperScript.path,
            "--source-root", sourceRoot.path,
            "--artifacts", artifactRoot.path,
            "--model-path", modelPath.path,
            "--rhubarb-bin", rhubarbExecutable.path,
            "--a2f-runner", a2fRunner.path,
            "--a2f-assets", a2fAssetDirectory.path,
            "--viewer-vendor", viewerVendorDirectory.path,
        ]
    }

    public func diagnostics(fileManager: FileManager = .default) -> [RuntimeDiagnostic] {
        var values = [RuntimeDiagnostic]()
        func append(_ label: String, _ url: URL, directory: Bool = false, required: Bool = true) {
            var isDirectory: ObjCBool = false
            let exists = fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
            let kindMatches = exists && (directory ? isDirectory.boolValue : !isDirectory.boolValue)
            values.append(
                RuntimeDiagnostic(
                    label: label,
                    path: url.path,
                    ready: kindMatches,
                    required: required
                )
            )
        }
        append("Source checkout", sourceRoot, directory: true)
        append("Python runtime", pythonExecutable)
        append("AutoAnim CLI", cliExecutable)
        append("Authenticated launcher", helperScript)
        append("MediaPipe face model", modelPath)
        append("Rhubarb", rhubarbExecutable)
        append("A2F runner", a2fRunner, required: false)
        append("A2F assets", a2fAssetDirectory, directory: true, required: false)
        append("Offline viewer", viewerVendorDirectory, directory: true)
        return values
    }

    public func validate(fileManager: FileManager = .default) throws {
        let missing = diagnostics(fileManager: fileManager).filter { $0.required && !$0.ready }
        guard missing.isEmpty else {
            throw RuntimeConfigurationError.missingRequiredPaths(missing)
        }
    }
}

public struct RuntimeDiagnostic: Equatable, Identifiable, Sendable {
    public var id: String { label }
    public let label: String
    public let path: String
    public let ready: Bool
    public let required: Bool

    public init(label: String, path: String, ready: Bool, required: Bool) {
        self.label = label
        self.path = path
        self.ready = ready
        self.required = required
    }
}

public enum RuntimeConfigurationError: LocalizedError, Equatable {
    case missingSourceRoot
    case missingBundleResources
    case missingRequiredPaths([RuntimeDiagnostic])

    public var errorDescription: String? {
        switch self {
        case .missingSourceRoot:
            return "This development build has no source checkout configured."
        case .missingBundleResources:
            return "The authenticated source-runtime launcher is missing from the app bundle."
        case .missingRequiredPaths(let paths):
            return "Source runtime is incomplete: \(paths.map(\.label).joined(separator: ", "))."
        }
    }
}
