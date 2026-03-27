import 'dart:typed_data';

/// Parses MIST binary audio frames from WebSocket binary messages.
///
/// Frame format (16-byte header + payload):
///   Bytes 0-3:   Magic (0x4D495354 = "MIST"), little-endian uint32
///   Bytes 4-5:   Message type (uint16): 0x01=chunk, 0x02=complete, 0x03=fade
///   Bytes 6-7:   Session ID (uint16)
///   Bytes 8-11:  Chunk sequence number (uint32)
///   Bytes 12-15: Sample rate (uint32)
///   Bytes 16+:   Raw PCM16 little-endian payload
class BinaryAudioFrame {
  static const int headerSize = 16;
  static const int _magic = 0x4D495354;

  static const int typeAudioChunk = 0x01;
  static const int typeAudioComplete = 0x02;
  static const int typeInterruptFade = 0x03;

  final int messageType;
  final int sessionId;
  final int chunkSeq;
  final int sampleRate;
  final Uint8List payload;

  BinaryAudioFrame._({
    required this.messageType,
    required this.sessionId,
    required this.chunkSeq,
    required this.sampleRate,
    required this.payload,
  });

  static BinaryAudioFrame? parse(Uint8List data) {
    if (data.length < headerSize) return null;

    final view = ByteData.view(data.buffer, data.offsetInBytes, data.length);
    final magic = view.getUint32(0, Endian.little);
    if (magic != _magic) return null;

    return BinaryAudioFrame._(
      messageType: view.getUint16(4, Endian.little),
      sessionId: view.getUint16(6, Endian.little),
      chunkSeq: view.getUint32(8, Endian.little),
      sampleRate: view.getUint32(12, Endian.little),
      payload: data.sublist(headerSize),
    );
  }

  bool get isAudioChunk => messageType == typeAudioChunk;
  bool get isAudioComplete => messageType == typeAudioComplete;
  bool get isInterruptFade => messageType == typeInterruptFade;
}
