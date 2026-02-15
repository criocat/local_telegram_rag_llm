from src.ingest import IndexedMessage, merge_messages_into_bursts, normalize_author, should_index_message


def test_normalize_author_with_username():
    assert normalize_author("nikita", 42) == "@nikita"


def test_normalize_author_without_username():
    assert normalize_author(None, 42) == "@tg_42"


def test_should_index_private_chat_always_true():
    assert should_index_message(is_private=True, sender_id=2, me_id=1, replied_to_sender_id=None)


def test_should_index_group_only_me_or_replies():
    assert should_index_message(is_private=False, sender_id=1, me_id=1, replied_to_sender_id=None)
    assert should_index_message(is_private=False, sender_id=2, me_id=1, replied_to_sender_id=1)
    assert not should_index_message(is_private=False, sender_id=2, me_id=1, replied_to_sender_id=3)


def test_merge_messages_same_author_within_60_sec():
    messages = [
        IndexedMessage(1, "chat", 10, 100, "@a", 1000, "hello", False),
        IndexedMessage(1, "chat", 11, 100, "@a", 1050, "world", False),
    ]
    bursts = merge_messages_into_bursts(messages)
    assert len(bursts) == 1
    assert bursts[0].message_ids == [10, 11]
    assert bursts[0].text == "hello\nworld"


def test_merge_messages_do_not_merge_replies():
    messages = [
        IndexedMessage(1, "chat", 10, 100, "@a", 1000, "hello", True),
        IndexedMessage(1, "chat", 11, 100, "@a", 1010, "world", False),
    ]
    bursts = merge_messages_into_bursts(messages)
    assert len(bursts) == 2
