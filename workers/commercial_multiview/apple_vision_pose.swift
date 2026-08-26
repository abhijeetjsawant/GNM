// Commercial macOS body-observation worker for AutoAnim.
//
// This executable intentionally emits only Apple Vision observations. Camera
// geometry, association, triangulation, temporal fitting, and retargeting live
// in AutoAnim and do not depend on research-only motion-capture software.

import CoreGraphics
import Foundation
import ImageIO
import Vision

struct JointSample: Codable {
    let x: Double
    let y: Double
    let confidence: Double
}

struct PersonSample: Codable {
    let index: Int
    let joints: [String: JointSample]
}

struct FrameSample: Codable {
    let schemaVersion: String
    let frameIndex: Int
    let imagePath: String
    let width: Int
    let height: Int
    let people: [PersonSample]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case frameIndex = "frame_index"
        case imagePath = "image_path"
        case width
        case height
        case people
    }
}

let jointNames: [(String, VNHumanBodyPoseObservation.JointName)] = [
    ("nose", .nose),
    ("neck", .neck),
    ("right_shoulder", .rightShoulder),
    ("right_elbow", .rightElbow),
    ("right_wrist", .rightWrist),
    ("left_shoulder", .leftShoulder),
    ("left_elbow", .leftElbow),
    ("left_wrist", .leftWrist),
    ("root", .root),
    ("right_hip", .rightHip),
    ("right_knee", .rightKnee),
    ("right_ankle", .rightAnkle),
    ("left_hip", .leftHip),
    ("left_knee", .leftKnee),
    ("left_ankle", .leftAnkle),
    ("right_eye", .rightEye),
    ("left_eye", .leftEye),
    ("right_ear", .rightEar),
    ("left_ear", .leftEar),
]

func imageSize(_ url: URL) throws -> (Int, Int) {
    guard
        let source = CGImageSourceCreateWithURL(url as CFURL, nil),
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
            as? [CFString: Any],
        let width = properties[kCGImagePropertyPixelWidth] as? Int,
        let height = properties[kCGImagePropertyPixelHeight] as? Int,
        width > 0,
        height > 0
    else {
        throw NSError(
            domain: "AutoAnimAppleVisionPose",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: "Unable to read image dimensions: \(url.path)"]
        )
    }
    return (width, height)
}

func frameIndex(_ url: URL, fallback: Int) -> Int {
    let stem = url.deletingPathExtension().lastPathComponent
    let digits = stem.reversed().prefix { $0.isNumber }.reversed()
    return Int(String(digits)) ?? fallback
}

func detect(_ path: String, fallbackIndex: Int) throws -> FrameSample {
    let url = URL(fileURLWithPath: path)
    let (width, height) = try imageSize(url)
    let request = VNDetectHumanBodyPoseRequest()
    let handler = VNImageRequestHandler(url: url, orientation: .up, options: [:])
    try handler.perform([request])

    var people: [PersonSample] = []
    for (personIndex, observation) in (request.results ?? []).enumerated() {
        let recognized = try observation.recognizedPoints(.all)
        var joints: [String: JointSample] = [:]
        for (name, visionName) in jointNames {
            guard let point = recognized[visionName] else { continue }
            joints[name] = JointSample(
                x: Double(point.location.x) * Double(width),
                y: (1.0 - Double(point.location.y)) * Double(height),
                confidence: Double(point.confidence)
            )
        }
        people.append(PersonSample(index: personIndex, joints: joints))
    }
    return FrameSample(
        schemaVersion: "autoanim.apple-vision-body-observations/1.0",
        frameIndex: frameIndex(url, fallback: fallbackIndex),
        imagePath: url.path,
        width: width,
        height: height,
        people: people
    )
}

let arguments = Array(CommandLine.arguments.dropFirst())
guard !arguments.isEmpty else {
    FileHandle.standardError.write(Data("usage: apple_vision_pose IMAGE...\n".utf8))
    exit(64)
}

let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
do {
    for (index, path) in arguments.enumerated() {
        let sample = try detect(path, fallbackIndex: index)
        FileHandle.standardOutput.write(try encoder.encode(sample))
        FileHandle.standardOutput.write(Data([0x0A]))
    }
} catch {
    FileHandle.standardError.write(Data("apple_vision_pose: \(error)\n".utf8))
    exit(1)
}
