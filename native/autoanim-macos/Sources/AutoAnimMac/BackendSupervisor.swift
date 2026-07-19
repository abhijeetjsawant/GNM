import AutoAnimMacCore
import Darwin
import Foundation

@MainActor
final class BackendSupervisor: ObservableObject {
    enum State: Equatable {
        case stopped
        case starting
        case ready
        case failed(String)

        var label: String {
            switch self {
            case .stopped: return "Stopped"
            case .starting: return "Starting source runtime…"
            case .ready: return "Ready"
            case .failed(let detail): return "Failed · \(detail)"
            }
        }
    }

    @Published private(set) var state: State = .stopped
    @Published private(set) var endpoint: LoopbackEndpoint?
    @Published private(set) var token: String?
    @Published private(set) var logLines: [String] = []

    let configuration: RuntimeConfiguration
    var onReady: ((LoopbackEndpoint, String) -> Void)?
    var onStopped: (() -> Void)?

    private var process: Process?
    private var standardOutput: Pipe?
    private var standardError: Pipe?
    private var outputBuffer = Data()

    init(configuration: RuntimeConfiguration) {
        self.configuration = configuration
    }

    func start() {
        guard process == nil else { return }
        do {
            try configuration.validate()
            let sessionToken = try SessionToken.generate()
            let child = Process()
            let stdout = Pipe()
            let stderr = Pipe()
            child.executableURL = configuration.pythonExecutable
            child.currentDirectoryURL = configuration.sourceRoot
            child.arguments = configuration.launchArguments + [
                "--session-token", sessionToken,
                "--native-parent-pid", String(Darwin.getpid()),
            ]
            var environment = ProcessInfo.processInfo.environment
            environment["PYTHONUNBUFFERED"] = "1"
            environment["AUTOANIM_CACHE_DIR"] = configuration.sourceRoot
                .appending(path: ".cache/autoanim_gnm").path
            child.environment = environment
            child.standardOutput = stdout
            child.standardError = stderr
            child.terminationHandler = { [weak self, weak child] finished in
                let status = finished.terminationStatus
                DispatchQueue.main.async {
                    guard let self, self.process === child else { return }
                    self.finishTermination(status: status)
                }
            }
            stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
                let data = handle.availableData
                guard !data.isEmpty else { return }
                DispatchQueue.main.async { self?.consumeStandardOutput(data) }
            }
            stderr.fileHandleForReading.readabilityHandler = { [weak self] handle in
                let data = handle.availableData
                guard !data.isEmpty else { return }
                let value = String(decoding: data, as: UTF8.self)
                DispatchQueue.main.async { self?.appendLog(value) }
            }
            state = .starting
            token = sessionToken
            process = child
            standardOutput = stdout
            standardError = stderr
            try child.run()
        } catch {
            cleanupHandles()
            process = nil
            token = nil
            state = .failed(error.localizedDescription)
            appendLog(error.localizedDescription)
        }
    }

    func restart() {
        stop()
        state = .starting
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { [weak self] in
            self?.state = .stopped
            self?.start()
        }
    }

    func stop() {
        guard let child = process else {
            endpoint = nil
            token = nil
            state = .stopped
            return
        }
        child.terminationHandler = nil
        if child.isRunning {
            child.terminate()
            let deadline = Date().addingTimeInterval(1.5)
            while child.isRunning && Date() < deadline {
                RunLoop.current.run(until: Date().addingTimeInterval(0.02))
            }
            if child.isRunning {
                let exactPID = child.processIdentifier
                if exactPID > 1 {
                    Darwin.kill(exactPID, SIGKILL)
                }
            }
            child.waitUntilExit()
        }
        cleanupHandles()
        process = nil
        endpoint = nil
        token = nil
        state = .stopped
        onStopped?()
    }

    private func consumeStandardOutput(_ data: Data) {
        outputBuffer.append(data)
        while let newline = outputBuffer.firstIndex(of: 0x0A) {
            let lineData = outputBuffer[..<newline]
            outputBuffer.removeSubrange(...newline)
            let line = String(decoding: lineData, as: UTF8.self)
            guard !line.isEmpty else { continue }
            if state == .starting {
                do {
                    let ready = try ServiceDecoding.readyEvent(from: line)
                    guard ready.event == "ready", ready.sourceRuntimeDependent else {
                        throw BackendSupervisorError.invalidHandshake
                    }
                    let validated = try LoopbackEndpoint(validating: ready.url)
                    guard let token else { throw BackendSupervisorError.invalidHandshake }
                    endpoint = validated
                    state = .ready
                    appendLog("Authenticated backend ready at \(validated.baseURL.absoluteString)")
                    onReady?(validated, token)
                    continue
                } catch {
                    appendLog("Ignored pre-ready output: \(line)")
                    continue
                }
            }
            appendLog(line)
        }
    }

    private func appendLog(_ value: String) {
        let lines = value.split(whereSeparator: \Character.isNewline).map(String.init)
        logLines.append(contentsOf: lines)
        if logLines.count > 120 {
            logLines.removeFirst(logLines.count - 120)
        }
    }

    private func finishTermination(status: Int32) {
        cleanupHandles()
        process = nil
        endpoint = nil
        token = nil
        if case .stopped = state {
            return
        }
        state = .failed("backend exited with status \(status)")
        onStopped?()
    }

    private func cleanupHandles() {
        standardOutput?.fileHandleForReading.readabilityHandler = nil
        standardError?.fileHandleForReading.readabilityHandler = nil
        standardOutput = nil
        standardError = nil
        outputBuffer.removeAll(keepingCapacity: true)
    }
}

enum BackendSupervisorError: LocalizedError {
    case invalidHandshake

    var errorDescription: String? {
        "The source runtime returned an invalid authenticated-loopback handshake."
    }
}
