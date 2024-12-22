import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict

import ccxt
import pandas as pd
from loguru import logger
from swarms import Agent

# Configure logging
logger.add("swarm.log", rotation="500 MB", level="INFO")


@dataclass
class SharedMemory:
    """Shared memory system for all agents"""

    market_data: Dict = None
    trading_signals: Dict = None
    risk_metrics: Dict = None
    positions: Dict = None

    def update(self, key: str, value: Any):
        """Update any attribute in shared memory"""
        if hasattr(self, key):
            setattr(self, key, value)
            logger.info(f"Updated shared memory: {key}")


class MarketDataAgent(Agent):
    def __init__(self, shared_memory: SharedMemory):
        super().__init__(
            agent_name="Market-Data-Agent",
            system_prompt="You are a market data specialist. Monitor and analyze crypto market data.",
        )
        self.exchange = ccxt.kraken({"enableRateLimit": True})
        self.shared_memory = shared_memory

    async def run(self, symbol: str) -> None:
        try:
            data = await self.exchange.fetch_ohlcv(
                symbol, "1h", limit=100
            )
            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )
            self.shared_memory.update("market_data", {symbol: df})

        except Exception as e:
            logger.error(f"Market data error: {e}")
            return None


class SignalAgent(Agent):
    def __init__(self, shared_memory: SharedMemory):
        super().__init__(
            agent_name="Signal-Agent",
            system_prompt="You are a trading signal specialist. Generate trading signals based on market data.",
        )
        self.shared_memory = shared_memory

    async def run(self, symbol: str) -> None:
        try:
            if not self.shared_memory.market_data:
                return

            df = self.shared_memory.market_data[symbol]

            # Get AI analysis
            analysis = await self.run(
                f"Analyze price action for {symbol}: Current price {df['close'].iloc[-1]}, "
                f"Volume: {df['volume'].iloc[-1]}"
            )

            signal = {
                "symbol": symbol,
                "action": analysis.get("recommendation", "HOLD"),
                "confidence": analysis.get("confidence", 0),
                "timestamp": pd.Timestamp.now(),
            }

            self.shared_memory.update(
                "trading_signals", {symbol: signal}
            )

        except Exception as e:
            logger.error(f"Signal generation error: {e}")


class RiskAgent(Agent):
    def __init__(self, shared_memory: SharedMemory):
        super().__init__(
            agent_name="Risk-Agent",
            system_prompt="You are a risk management specialist. Evaluate trading risks and set position sizes.",
        )
        self.shared_memory = shared_memory

    async def run(self, symbol: str) -> None:
        try:
            if not (
                self.shared_memory.market_data
                and self.shared_memory.trading_signals
            ):
                return

            signal = self.shared_memory.trading_signals[symbol]

            risk_metrics = {
                "symbol": symbol,
                "position_size": Decimal(
                    "0.01"
                ),  # Default conservative size
                "risk_score": 0.5,
                "timestamp": pd.Timestamp.now(),
            }

            # Get AI risk assessment
            assessment = await self.run(
                f"Evaluate risk for {symbol} trade with signal confidence {signal['confidence']}"
            )

            risk_metrics.update(assessment)
            self.shared_memory.update(
                "risk_metrics", {symbol: risk_metrics}
            )

        except Exception as e:
            logger.error(f"Risk assessment error: {e}")


class QueenAgent(Agent):
    def __init__(self, shared_memory: SharedMemory):
        super().__init__(
            agent_name="Queen-Agent",
            system_prompt="You are the queen bee coordinator. Make final trading decisions based on all available information.",
        )
        self.shared_memory = shared_memory
        self.market_agent = MarketDataAgent(shared_memory)
        self.signal_agent = SignalAgent(shared_memory)
        self.risk_agent = RiskAgent(shared_memory)

    async def run(self, symbol: str) -> Dict:
        """Coordinate the swarm and make final decisions"""
        try:
            # Run all agents sequentially
            await self.market_agent.run(symbol)
            await self.signal_agent.run(symbol)
            await self.risk_agent.run(symbol)

            # Make final decision
            if all(
                [
                    self.shared_memory.market_data,
                    self.shared_memory.trading_signals,
                    self.shared_memory.risk_metrics,
                ]
            ):
                signal = self.shared_memory.trading_signals[symbol]
                risk = self.shared_memory.risk_metrics[symbol]

                # Get AI final decision
                decision = await self.run(
                    f"Make final trading decision for {symbol}:\n"
                    f"Signal: {signal['action']}\n"
                    f"Confidence: {signal['confidence']}\n"
                    f"Risk Score: {risk['risk_score']}"
                )

                # Update positions if trade is approved
                if decision.get("execute", False):
                    self.shared_memory.update(
                        "positions",
                        {
                            symbol: {
                                "action": signal["action"],
                                "size": risk["position_size"],
                                "timestamp": pd.Timestamp.now(),
                            }
                        },
                    )

                return decision

        except Exception as e:
            logger.error(f"Queen agent error: {e}")
            return {"execute": False, "reason": str(e)}


async def main():
    # Initialize shared memory
    shared_memory = SharedMemory()

    # Initialize queen agent
    queen = QueenAgent(shared_memory)

    symbols = ["BTC/USD", "ETH/USD"]

    try:
        while True:
            for symbol in symbols:
                decision = await queen.run(symbol)
                logger.info(f"Decision for {symbol}: {decision}")
                await asyncio.sleep(1)  # Rate limiting

            await asyncio.sleep(60)  # Main loop interval

    except KeyboardInterrupt:
        logger.info("Shutting down swarm...")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
