from riichienv import convert


class TestConvert:
    def test_tid_to_mpsz(self):
        # 1m -> 0
        assert convert.tid_to_mpsz(0) == "1m"
        # 5m Red -> 16
        assert convert.tid_to_mpsz(16) == "0m"
        # 5m Non-red -> 17
        assert convert.tid_to_mpsz(17) == "5m"
        # 9m -> 32
        assert convert.tid_to_mpsz(32) == "9m"

        # 1z (East) -> 108
        assert convert.tid_to_mpsz(108) == "1z"
        # 5z (White) -> 108 + 16 = 124
        assert convert.tid_to_mpsz(124) == "5z"

    def test_tid_to_mjai(self):
        # 1m -> 1m
        assert convert.tid_to_mjai(0) == "1m"
        # 5m Red -> 5mr
        assert convert.tid_to_mjai(16) == "5mr"
        # 5m Non-red -> 5m
        assert convert.tid_to_mjai(17) == "5m"

        # 1z -> E
        assert convert.tid_to_mjai(108) == "E"
        # 2z -> S
        assert convert.tid_to_mjai(112) == "S"
        # 5z -> P
        assert convert.tid_to_mjai(124) == "P"
        # 6z -> F
        assert convert.tid_to_mjai(128) == "F"
        # 7z -> C
        assert convert.tid_to_mjai(132) == "C"

    def test_mpsz_to_tid(self):
        assert convert.mpsz_to_tid("1m") == 0
        assert convert.mpsz_to_tid("0m") == 16
        assert convert.mpsz_to_tid("5m") == 17  # Canonical non-red
        assert convert.mpsz_to_tid("1z") == 108
        assert convert.mpsz_to_tid("5z") == 124

    def test_mjai_to_tid(self):
        assert convert.mjai_to_tid("1m") == 0
        assert convert.mjai_to_tid("5mr") == 16
        assert convert.mjai_to_tid("5m") == 17
        assert convert.mjai_to_tid("E") == 108
        assert convert.mjai_to_tid("P") == 124
        assert convert.mjai_to_tid("C") == 132

    def test_conversions_cross(self):
        # mjai -> mpsz
        assert convert.mjai_to_mpsz("E") == "1z"
        assert convert.mjai_to_mpsz("5mr") == "0m"

        # mpsz -> mjai
        assert convert.mpsz_to_mjai("1z") == "E"
        assert convert.mpsz_to_mjai("0p") == "5pr"

    def test_lists(self):
        tids = [0, 16, 124]
        assert convert.tid_to_mpsz_list(tids) == ["1m", "0m", "5z"]
        assert convert.tid_to_mjai_list(tids) == ["1m", "5mr", "P"]
