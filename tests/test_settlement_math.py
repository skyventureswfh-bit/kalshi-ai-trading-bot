import pytest

from src.auto.settlement_math import FinalMinuteAccumulator, UnsafeSettlementData


def test_required_remaining_average_and_flat_projection():
    item = FinalMinuteAccumulator(window_start_epoch=1000, target=100.0, window_seconds=4)
    item.add(event_epoch=1001.1, value=98.0, received_epoch=1001.2)
    item.add(event_epoch=1002.1, value=100.0, received_epoch=1002.2)
    result = item.projection()
    assert result.observed_seconds == 2
    assert result.remaining_seconds == 2
    assert result.observed_average == 99.0
    assert result.projected_average_if_flat == 99.5
    assert result.required_remaining_average == 101.0
    assert result.projected_margin == -0.5


def test_complete_window_returns_exact_average():
    item = FinalMinuteAccumulator(window_start_epoch=2000, target=100.0, window_seconds=3)
    for offset, value in enumerate((99.0, 100.0, 104.0)):
        item.add(event_epoch=2001 + offset, value=value, received_epoch=2001 + offset + 0.1)
    assert item.final_average() == pytest.approx(101.0)


def test_incomplete_window_fails_closed():
    item = FinalMinuteAccumulator(window_start_epoch=3000, target=100.0, window_seconds=3)
    item.add(event_epoch=3001, value=100.0, received_epoch=3001.1)
    with pytest.raises(UnsafeSettlementData, match="incomplete"):
        item.final_average()


def test_stale_and_out_of_order_samples_fail_closed():
    item = FinalMinuteAccumulator(window_start_epoch=4000, target=100.0, window_seconds=3)
    with pytest.raises(UnsafeSettlementData, match="stale"):
        item.add(event_epoch=4001, value=100.0, received_epoch=4004)
    item.add(event_epoch=4002, value=100.0, received_epoch=4002.1)
    with pytest.raises(UnsafeSettlementData, match="out-of-order"):
        item.add(event_epoch=4001, value=100.0, received_epoch=4001.1)

