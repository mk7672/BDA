
import json
from datetime import datetime, timezone
from urllib.parse import quote, urlsplit, urlunsplit

import requests
from kafka import KafkaConsumer

KAFKA_SERVER = "localhost:9092"
TOPIC = "customer-reviews"
HDFS_URL = "http://localhost:9870"
HDFS_DIRECTORY = "/reviews"


def save_review_to_hdfs(review):
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = f"{HDFS_DIRECTORY}/reviews-{date}.jsonl"
    encoded_path = quote(path, safe="/")
    url = f"{HDFS_URL}/webhdfs/v1{encoded_path}"
    review_line = (
        json.dumps(review, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    # Try to create today's JSONL file.
    response = requests.put(
        url,
        params={"op": "CREATE", "overwrite": "false"},
        allow_redirects=False,
        timeout=15,
    )

    if response.status_code in (307, 308):
        upload_url = response.headers["Location"]

        # Docker's "datanode" hostname may not resolve on Windows.
        parts = urlsplit(upload_url)
        if parts.hostname == "datanode":
            upload_url = urlunsplit((
                parts.scheme,
                f"localhost:{parts.port}" if parts.port else "localhost",
                parts.path,
                parts.query,
                parts.fragment,
            ))

        upload = requests.put(
            upload_url,
            data=review_line,
            headers={"Content-Type": "application/octet-stream"},
            timeout=15,
        )
        upload.raise_for_status()
        return path

    # The file may already exist, so append this review.
    if response.status_code not in (200, 201):
        response.raise_for_status()
        raise RuntimeError(
            f"Unexpected HDFS CREATE response: "
            f"{response.status_code} {response.text}"
        )

    append_response = requests.post(
        url,
        params={"op": "APPEND"},
        allow_redirects=False,
        timeout=15,
    )
    append_response.raise_for_status()

    if append_response.status_code in (307, 308):
        upload_url = append_response.headers["Location"]

        parts = urlsplit(upload_url)
        if parts.hostname == "datanode":
            upload_url = urlunsplit((
                parts.scheme,
                f"localhost:{parts.port}" if parts.port else "localhost",
                parts.path,
                parts.query,
                parts.fragment,
            ))

        upload = requests.post(
            upload_url,
            data=review_line,
            headers={"Content-Type": "application/octet-stream"},
            timeout=15,
        )
        upload.raise_for_status()

    return path


def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_SERVER,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id="bda-review-consumer",
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        ),
    )

    print(f"Listening for reviews on '{TOPIC}'... Press Ctrl+C to stop.")

    try:
        for message in consumer:
            review = message.value
            print("Received review:")
            print(json.dumps(review, indent=2))

            saved_path = save_review_to_hdfs(review)
            print(f"Saved to HDFS: {saved_path}")

            # Commit only after the HDFS write succeeds.
            consumer.commit()

    except KeyboardInterrupt:
        print("\nConsumer stopped.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
