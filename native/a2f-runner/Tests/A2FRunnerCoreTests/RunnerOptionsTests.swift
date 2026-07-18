import Foundation
import XCTest
@testable import A2FRunnerCore

final class RunnerOptionsTests: XCTestCase {
    func testParsesRequiredAndModelOptions() throws {
        let options = try RunnerOptions.parse([
            "--input", "speech.wav",
            "--output", "motion.jsonl",
            "--model", "owner/model",
            "--emotion", "joy",
            "--emotion-strength", "0.65",
            "--offline",
            "--verbose"
        ])

        XCTAssertEqual(options.input, "speech.wav")
        XCTAssertEqual(options.output, "motion.jsonl")
        XCTAssertEqual(options.model, "owner/model")
        XCTAssertNil(options.modelDirectory)
        XCTAssertTrue(options.offline)
        XCTAssertTrue(options.verbose)
        XCTAssertEqual(options.emotion, "joy")
        XCTAssertEqual(options.emotionStrength, 0.65)
    }

    func testUsesPinnedReleaseDefaultModel() throws {
        let options = try RunnerOptions.parse([
            "-i", "speech.wav", "-o", "motion.jsonl"
        ])

        XCTAssertEqual(
            options.model,
            "aufklarer/Audio2Face-3D-v2.3.1-Claire-MLX"
        )
    }

    func testModelAndModelDirectoryAreMutuallyExclusive() {
        XCTAssertThrowsError(
            try RunnerOptions.parse([
                "--input", "speech.wav",
                "--output", "motion.jsonl",
                "--model", "owner/model",
                "--model-dir", "/models/local"
            ])
        ) { error in
            XCTAssertEqual(
                error as? RunnerArgumentError,
                .mutuallyExclusive("--model", "--model-dir")
            )
        }
    }

    func testMissingValueAndUnknownOptionAreTyped() {
        XCTAssertThrowsError(try RunnerOptions.parse(["--input"])) { error in
            XCTAssertEqual(error as? RunnerArgumentError, .missingValue("--input"))
        }
        XCTAssertThrowsError(try RunnerOptions.parse(["--wat"])) { error in
            XCTAssertEqual(error as? RunnerArgumentError, .unknownOption("--wat"))
        }
    }

    func testEmotionValidationIsTyped() {
        XCTAssertThrowsError(
            try RunnerOptions.parse([
                "--input", "speech.wav", "--output", "motion.jsonl",
                "--emotion", "confused",
            ])
        ) { error in
            XCTAssertEqual(error as? RunnerArgumentError, .invalidEmotion("confused"))
        }
        XCTAssertThrowsError(
            try RunnerOptions.parse([
                "--input", "speech.wav", "--output", "motion.jsonl",
                "--emotion-strength", "1.5",
            ])
        ) { error in
            XCTAssertEqual(error as? RunnerArgumentError, .invalidEmotionStrength("1.5"))
        }
    }

    func testMissingRequiredOptionIsTyped() {
        XCTAssertThrowsError(
            try RunnerOptions.parse(["--input", "speech.wav"])
        ) { error in
            XCTAssertEqual(
                error as? RunnerArgumentError,
                .missingRequiredOption("--output")
            )
        }
    }

    func testValidatesExtensionsAndExistingInput() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("A2FRunnerCoreTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let input = directory.appendingPathComponent("speech.wav")
        try Data().write(to: input)
        let valid = RunnerOptions(
            input: input.path,
            output: directory.appendingPathComponent("motion.jsonl").path
        )
        XCTAssertNoThrow(try valid.validatePaths())

        let invalid = RunnerOptions(
            input: input.path,
            output: directory.appendingPathComponent("motion.json").path
        )
        XCTAssertThrowsError(try invalid.validatePaths()) { error in
            XCTAssertEqual(
                error as? RunnerArgumentError,
                .invalidOutputExtension("json")
            )
        }
    }
}
