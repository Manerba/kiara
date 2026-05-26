from kiara.web.api_utils import lazy_singleton


def test_lazy_singleton_caches_none_return_value():
    calls = {"count": 0}

    @lazy_singleton
    def get_value():
        calls["count"] += 1
        return None

    assert get_value() is None
    assert get_value() is None
    assert calls["count"] == 1

    get_value.reset()
    assert get_value() is None
    assert calls["count"] == 2
