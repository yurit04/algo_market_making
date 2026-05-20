import numpy as np
import pytest

from algo_mm.models.avellaneda_stoikov import AvellanedaStoikovMarketMaker
from algo_mm.models.inventory_adjusted import InventoryAdjustedMarketMaker
from algo_mm.models.naive import NaiveMarketMaker


class TestNaiveMarketMaker:
    def setup_method(self):
        self.model = NaiveMarketMaker(half_spread=0.1, lambda_buy=100.0, lambda_sell=100.0)

    def test_symmetric_quotes(self):
        bid, ask = self.model.compute_quotes(mid=100.0, inventory=0, t=0.0)
        assert abs(bid - 99.9) < 1e-9
        assert abs(ask - 100.1) < 1e-9

    def test_spread_width(self):
        bid, ask = self.model.compute_quotes(100.0, inventory=0, t=0.0)
        assert abs((ask - bid) - 2 * self.model.half_spread) < 1e-9

    def test_ignores_inventory(self):
        bid0, ask0 = self.model.compute_quotes(100.0, inventory=0, t=0.0)
        bid5, ask5 = self.model.compute_quotes(100.0, inventory=5, t=0.0)
        assert bid0 == bid5
        assert ask0 == ask5

    def test_fill_probs_in_unit_interval(self):
        bid, ask = self.model.compute_quotes(100.0, 0, 0.0)
        prob_bid, prob_ask = self.model.fill_probs(100.0, bid, ask, dt=0.005)
        assert 0 < prob_bid < 1
        assert 0 < prob_ask < 1


class TestInventoryAdjustedMarketMaker:
    def setup_method(self):
        self.model = InventoryAdjustedMarketMaker(
            base_spread=0.2, skew=0.05, q_target=0,
            lambda_buy=100.0, lambda_sell=100.0,
        )

    def test_spread_constant_across_inventory(self):
        for q in [-3, 0, 3]:
            bid, ask = self.model.compute_quotes(100.0, inventory=q, t=0.0)
            assert abs((ask - bid) - self.model.base_spread) < 1e-9

    def test_long_inventory_shifts_quotes_up(self):
        bid_long, ask_long = self.model.compute_quotes(100.0, inventory=2, t=0.0)
        bid_flat, ask_flat = self.model.compute_quotes(100.0, inventory=0, t=0.0)
        assert bid_long > bid_flat
        assert ask_long > ask_flat

    def test_short_inventory_shifts_quotes_down(self):
        bid_short, ask_short = self.model.compute_quotes(100.0, inventory=-2, t=0.0)
        bid_flat, ask_flat = self.model.compute_quotes(100.0, inventory=0, t=0.0)
        assert bid_short < bid_flat
        assert ask_short < ask_flat


class TestAvellanedaStoikovMarketMaker:
    def setup_method(self):
        self.model = AvellanedaStoikovMarketMaker(
            sigma=2.0, gamma=0.1, kappa=1.5, A=140.0, T=1.0,
        )

    def test_zero_inventory_quotes_symmetric_around_mid(self):
        mid = 100.0
        bid, ask = self.model.compute_quotes(mid, inventory=0, t=0.5)
        assert abs((bid + ask) / 2 - mid) < 1e-9

    def test_positive_inventory_skews_quotes_down(self):
        mid = 100.0
        bid_long, ask_long = self.model.compute_quotes(mid, inventory=3, t=0.5)
        bid_flat, ask_flat = self.model.compute_quotes(mid, inventory=0, t=0.5)
        assert bid_long < bid_flat
        assert ask_long < ask_flat

    def test_spread_widens_as_time_remaining_increases(self):
        mid, q = 100.0, 0
        bid_early, ask_early = self.model.compute_quotes(mid, q, t=0.1)
        bid_late, ask_late = self.model.compute_quotes(mid, q, t=0.9)
        spread_early = ask_early - bid_early
        spread_late = ask_late - bid_late
        assert spread_early > spread_late

    def test_fill_probs_in_unit_interval(self):
        bid, ask = self.model.compute_quotes(100.0, 0, 0.5)
        prob_bid, prob_ask = self.model.fill_probs(100.0, bid, ask, dt=0.005)
        assert 0 < prob_bid < 1
        assert 0 < prob_ask < 1
