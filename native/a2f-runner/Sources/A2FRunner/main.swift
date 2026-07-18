import A2FRunnerCore
import Audio2Face3D
import AudioCommon
import Darwin
import Foundation

@main
struct A2FRunner {
    static func main() async {
        do {
            let options = try RunnerOptions.parse(Array(CommandLine.arguments.dropFirst()))
            try options.validatePaths()
            try await run(options)
        } catch RunnerArgumentError.helpRequested {
            print(runnerUsage)
        } catch let error as RunnerArgumentError {
            writeStandardError("error: \(error.localizedDescription)\n\n\(runnerUsage)\n")
            exit(2)
        } catch {
            writeStandardError("error: \(error.localizedDescription)\n")
            exit(1)
        }
    }

    private static func run(_ options: RunnerOptions) async throws {
        let inputURL = URL(fileURLWithPath: options.input)
        let outputURL = URL(fileURLWithPath: options.output)
        let audio = try AudioFileLoader.load(
            url: inputURL,
            targetSampleRate: 16_000,
            quality: .standard
        )

        let model: Audio2Face3DModel
        if let modelDirectory = options.modelDirectory {
            model = try Audio2Face3DModel.fromLocal(
                directory: URL(fileURLWithPath: modelDirectory, isDirectory: true)
            )
        } else {
            model = try await Audio2Face3DModel.fromPretrained(
                modelId: options.model,
                offlineMode: options.offline
            ) { progress, message in
                guard options.verbose else { return }
                writeStandardError(String(format: "[%3.0f%%] %@\n", progress * 100, message))
            }
        }

        let emotionOrder = [
            "surprise", "anger", "contempt", "disgust", "fear",
            "grief", "joy", "outofbreath", "pain", "sad",
        ]
        var explicitEmotion = [Float](repeating: 0, count: emotionOrder.count)
        if let index = emotionOrder.firstIndex(of: options.emotion) {
            explicitEmotion[index] = options.emotionStrength
        }
        let emotionVector = try model.emotionVector(explicit: explicitEmotion)
        let start = CFAbsoluteTimeGetCurrent()
        let frames = try model.frames(
            for: audio,
            sampleRate: 16_000,
            emotion: emotionVector
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var payload = Data()
        for frame in frames {
            payload.append(try encoder.encode(frame))
            payload.append(0x0A)
        }

        let outputDirectory = outputURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )
        try payload.write(to: outputURL, options: .atomic)

        if options.verbose {
            let duration = Double(audio.count) / 16_000.0
            let layout = model.configuration.coefficientLayout
            writeStandardError(
                String(
                    format: "model=%@ emotion=%@ strength=%.2f audio=%.3fs frames=%d coefficients=%d (skin=%d tongue=%d jaw=%d eyes=%d) inference=%.3fs\n",
                    model.configuration.modelId,
                    options.emotion,
                    options.emotionStrength,
                    duration,
                    frames.count,
                    layout.coefficientCount,
                    layout.skinCount,
                    layout.tongueCount,
                    layout.jawCount,
                    layout.eyeCount,
                    elapsed
                )
            )
        }
        print(outputURL.path)
    }

    private static func writeStandardError(_ message: String) {
        FileHandle.standardError.write(Data(message.utf8))
    }
}
