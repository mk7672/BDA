
import json
from datetime import datetime, timezone

from kafka import KafkaProducer

KAFKA_SERVER = "localhost:9092"
TOPIC = "customer-reviews"


def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=5,
    )


def send_review(producer, review):
    future = producer.send(TOPIC, value=review)
    metadata = future.get(timeout=10)

    print(
        f"Sent review {review['review_id']} "
        f"to topic={metadata.topic}, partition={metadata.partition}, "
        f"offset={metadata.offset}"
    )


def main():
    review = {
        "review_id": "R001",
        "product": "Wireless Headphones",
        "review_text": "The battery stopped working after two days",
        "rating": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    producer = create_producer()

    try:
        send_review(producer, review)
    finally:
        producer.close()


if __name__ == "__main__":
    main()
