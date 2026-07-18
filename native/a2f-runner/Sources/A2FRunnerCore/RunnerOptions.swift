import Foundation

public struct RunnerOptions: Equatable, Sendable {
    public static let defaultModel = "aufklarer/Audio2Face-3D-v2.3.1-Claire-MLX"
    public static let supportedEmotions = [
        "neutral", "surprise", "anger", "contempt", "disgust",
        "fear", "grief", "joy", "outofbreath", "pain", "sad",
    ]

    public let input: String
    public let output: String
    public let model: String
    public let modelDirectory: String?
    public let offline: Bool
    public let verbose: Bool
    public let emotion: String
    public let emotionStrength: Float

    public init(
        input: String,
        output: String,
        model: String = Self.defaultModel,
        modelDirectory: String? = nil,
        offline: Bool = false,
        verbose: Bool = false,
        emotion: String = "neutral",
        emotionStrength: Float = 1.0
    ) {
        self.input = input
        self.output = output
        self.model = model
        self.modelDirectory = modelDirectory
        self.offline = offline
        self.verbose = verbose
        self.emotion = emotion
        self.emotionStrength = emotionStrength
    }

    public static func parse(_ arguments: [String]) throws -> RunnerOptions {
        var input: String?
        var output: String?
        var model: String?
        var modelDirectory: String?
        var offline = false
        var verbose = false
        var emotion: String?
        var emotionStrength: Float?
        var index = 0

        func value(after flag: String) throws -> String {
            let valueIndex = index + 1
            guard valueIndex < arguments.count, !arguments[valueIndex].hasPrefix("-") else {
                throw RunnerArgumentError.missingValue(flag)
            }
            return arguments[valueIndex]
        }

        func assign(_ destination: inout String?, value: String, flag: String) throws {
            guard destination == nil else {
                throw RunnerArgumentError.duplicateOption(flag)
            }
            destination = value
        }

        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--help", "-h":
                throw RunnerArgumentError.helpRequested
            case "--input", "-i":
                try assign(&input, value: value(after: argument), flag: "--input")
                index += 2
            case "--output", "-o":
                try assign(&output, value: value(after: argument), flag: "--output")
                index += 2
            case "--model":
                try assign(&model, value: value(after: argument), flag: "--model")
                index += 2
            case "--model-dir":
                try assign(&modelDirectory, value: value(after: argument), flag: "--model-dir")
                index += 2
            case "--offline":
                guard !offline else { throw RunnerArgumentError.duplicateOption("--offline") }
                offline = true
                index += 1
            case "--verbose", "-v":
                guard !verbose else { throw RunnerArgumentError.duplicateOption("--verbose") }
                verbose = true
                index += 1
            case "--emotion":
                try assign(&emotion, value: value(after: argument), flag: "--emotion")
                index += 2
            case "--emotion-strength":
                let raw = try value(after: argument)
                guard emotionStrength == nil else {
                    throw RunnerArgumentError.duplicateOption("--emotion-strength")
                }
                guard let parsed = Float(raw), parsed.isFinite, (0.0...1.0).contains(parsed) else {
                    throw RunnerArgumentError.invalidEmotionStrength(raw)
                }
                emotionStrength = parsed
                index += 2
            default:
                if argument.hasPrefix("-") {
                    throw RunnerArgumentError.unknownOption(argument)
                }
                throw RunnerArgumentError.unexpectedArgument(argument)
            }
        }

        guard let input else { throw RunnerArgumentError.missingRequiredOption("--input") }
        guard let output else { throw RunnerArgumentError.missingRequiredOption("--output") }
        if model != nil, modelDirectory != nil {
            throw RunnerArgumentError.mutuallyExclusive("--model", "--model-dir")
        }
        let selectedEmotion = emotion?.lowercased() ?? "neutral"
        guard supportedEmotions.contains(selectedEmotion) else {
            throw RunnerArgumentError.invalidEmotion(selectedEmotion)
        }

        return RunnerOptions(
            input: input,
            output: output,
            model: model ?? Self.defaultModel,
            modelDirectory: modelDirectory,
            offline: offline,
            verbose: verbose,
            emotion: selectedEmotion,
            emotionStrength: emotionStrength ?? 1.0
        )
    }

    public func validatePaths(fileManager: FileManager = .default) throws {
        let inputURL = URL(fileURLWithPath: input)
        guard inputURL.pathExtension.lowercased() == "wav" else {
            throw RunnerArgumentError.invalidInputExtension(inputURL.pathExtension)
        }
        guard URL(fileURLWithPath: output).pathExtension.lowercased() == "jsonl" else {
            throw RunnerArgumentError.invalidOutputExtension(
                URL(fileURLWithPath: output).pathExtension
            )
        }

        var inputIsDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: inputURL.path, isDirectory: &inputIsDirectory) else {
            throw RunnerArgumentError.inputNotFound(inputURL.path)
        }
        guard !inputIsDirectory.boolValue else {
            throw RunnerArgumentError.inputIsDirectory(inputURL.path)
        }

        if let modelDirectory {
            let modelURL = URL(fileURLWithPath: modelDirectory)
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: modelURL.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else {
                throw RunnerArgumentError.modelDirectoryNotFound(modelURL.path)
            }
        }

        if inputURL.standardizedFileURL == URL(fileURLWithPath: output).standardizedFileURL {
            throw RunnerArgumentError.inputEqualsOutput(inputURL.path)
        }
    }
}

public enum RunnerArgumentError: Error, Equatable, LocalizedError, Sendable {
    case helpRequested
    case missingValue(String)
    case duplicateOption(String)
    case unknownOption(String)
    case unexpectedArgument(String)
    case missingRequiredOption(String)
    case mutuallyExclusive(String, String)
    case invalidInputExtension(String)
    case invalidOutputExtension(String)
    case inputNotFound(String)
    case inputIsDirectory(String)
    case modelDirectoryNotFound(String)
    case inputEqualsOutput(String)
    case invalidEmotion(String)
    case invalidEmotionStrength(String)

    public var errorDescription: String? {
        switch self {
        case .helpRequested:
            return nil
        case .missingValue(let flag):
            return "Missing value for \(flag)."
        case .duplicateOption(let flag):
            return "Option \(flag) may only be provided once."
        case .unknownOption(let flag):
            return "Unknown option: \(flag)."
        case .unexpectedArgument(let argument):
            return "Unexpected positional argument: \(argument)."
        case .missingRequiredOption(let flag):
            return "Missing required option: \(flag)."
        case .mutuallyExclusive(let first, let second):
            return "Options \(first) and \(second) are mutually exclusive."
        case .invalidInputExtension(let extensionName):
            let suffix = extensionName.isEmpty ? "<none>" : extensionName
            return "Input must be a WAV file; got extension \(suffix)."
        case .invalidOutputExtension(let extensionName):
            let suffix = extensionName.isEmpty ? "<none>" : extensionName
            return "Output must use the .jsonl extension; got \(suffix)."
        case .inputNotFound(let path):
            return "Input WAV does not exist: \(path)."
        case .inputIsDirectory(let path):
            return "Input WAV is a directory: \(path)."
        case .modelDirectoryNotFound(let path):
            return "Model directory does not exist or is not a directory: \(path)."
        case .inputEqualsOutput(let path):
            return "Input and output paths must differ: \(path)."
        case .invalidEmotion(let emotion):
            return "Unsupported Audio2Face emotion: \(emotion)."
        case .invalidEmotionStrength(let value):
            return "Emotion strength must be a finite value in [0,1]; got \(value)."
        }
    }
}

public let runnerUsage = """
Usage:
  a2f-runner --input INPUT.wav --output OUTPUT.jsonl [options]

Required:
  -i, --input PATH       Input WAV file (decoded and resampled to 16 kHz mono)
  -o, --output PATH      Output JSON Lines file

Model selection (mutually exclusive):
      --model ID         Hugging Face model ID (default: \(RunnerOptions.defaultModel))
      --model-dir PATH   Local exported Audio2Face3D MLX bundle

Options:
      --offline          Forbid downloads when resolving --model
      --emotion NAME     Explicit acting direction (default: neutral)
      --emotion-strength N
                         Explicit emotion strength in [0,1] (default: 1)
  -v, --verbose          Print model, frame, and inference timing details
  -h, --help             Show this help
"""
