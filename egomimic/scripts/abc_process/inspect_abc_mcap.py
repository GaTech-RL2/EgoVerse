"""
Dump the channel/schema inventory of an ABC-130k episode.mcap.

Run this once before converting: the topic names and protobuf schema names it
prints are what abc_to_zarr.py's TOPIC_MAP has to be pinned against.

    python -m egomimic.scripts.abc_process.inspect_abc_mcap /path/to/episode.mcap
"""

import argparse
from collections import defaultdict

from mcap.reader import make_reader


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mcap_path")
    p.add_argument(
        "--sample",
        action="store_true",
        help="also decode and print one message per topic (needs mcap-protobuf-support)",
    )
    args = p.parse_args()

    with open(args.mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()

        print(f"library: {reader.get_header().library}")
        print(f"profile: {reader.get_header().profile}\n")

        counts = (
            summary.statistics.channel_message_counts if summary.statistics else {}
        )
        rows = []
        for chan_id, chan in summary.channels.items():
            schema = summary.schemas.get(chan.schema_id)
            rows.append(
                (
                    chan.topic,
                    schema.name if schema else "-",
                    schema.encoding if schema else "-",
                    counts.get(chan_id, 0),
                )
            )
        rows.sort()
        w = max(len(r[0]) for r in rows)
        print(f"{'TOPIC'.ljust(w)}  {'SCHEMA':<45} {'ENC':<10} COUNT")
        for topic, sname, enc, n in rows:
            print(f"{topic.ljust(w)}  {sname:<45} {enc:<10} {n}")

        if summary.statistics:
            s = summary.statistics
            span_ns = s.message_end_time - s.message_start_time
            print(f"\nmessages: {s.message_count}   span: {span_ns / 1e9:.2f}s")

    if not args.sample:
        return

    from mcap_protobuf.decoder import DecoderFactory

    print("\n--- first message per topic ---")
    seen = defaultdict(int)
    with open(args.mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, _msg, decoded in reader.iter_decoded_messages():
            if seen[channel.topic]:
                continue
            seen[channel.topic] += 1
            text = str(decoded)
            print(f"\n[{channel.topic}]\n{text[:600]}")


if __name__ == "__main__":
    main()
