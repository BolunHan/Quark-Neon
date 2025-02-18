import datetime
import pathlib

from quark.base import safe_exit
from quark.calibration.future import FutureTopic
from quark.profile import profile_cn_override

from factor.evaluation import FactorChain, FactorNode

START_DATE = datetime.date(2025, 1, 1)
END_DATE = datetime.date(2025, 1, 15)


def eval_individual():
    nodes = FactorNode.from_file(file=pathlib.Path(__file__).parent.joinpath('entropy.volume_profile.py'))

    for node in nodes:
        tree = FactorChain(node)

        tree.extend(
            nodes=FactorNode.from_file(
                file=pathlib.Path(__file__).parents[1].joinpath('_aux', '_aux.py'),
                names=[
                    'Monitor.Aux.Adaptive.Index.30',
                ]
            )
        )

        tree.evaluate(
            start_date=START_DATE,
            end_date=END_DATE,
            dtype=['TradeData'],
            pred_var=list(str(_) for _ in FutureTopic),
            # pred_var=[],
            index_name='000016.SH',
            pred_target='IH_MAIN',
            strategy_mode='sampling',
            dev_pool=False,
            override_cache=False,
            resume=True
        )


def eval_all():
    tree = FactorChain()

    # add entropy monitor
    tree.extend(
        nodes=FactorNode.from_file(
            file=pathlib.Path(__file__).parent.joinpath('entropy.volume_profile.py'),
            # names='Monitor.Entropy.VolumeProfile.15'
        )
    )

    # # add aux monitor
    tree.extend(
        nodes=FactorNode.from_file(
            file=pathlib.Path(__file__).parents[1].joinpath('_aux', '_aux.py'),
            names=[
                'Monitor.Aux.Adaptive.Index.15',
                'Monitor.Aux.Adaptive.Index.30',
                'Monitor.Aux.Adaptive.Index.60',
            ]
        )
    )

    tree.evaluate(
        start_date=START_DATE,
        end_date=END_DATE,
        # market_date=datetime.date(2024, 12, 1),
        dtype=['TradeData'],
        # pred_var=list(str(_) for _ in FutureTopic),
        pred_var=[],
        index_name='000016.SH',
        pred_target='IH_MAIN',
        strategy_mode='sampling',
        dev_pool=False,
        override_cache=True,
        resume=False
    )


def sim():
    tree = FactorChain(
        nodes=FactorNode.from_file(
            file=pathlib.Path(__file__).parent.joinpath('entropy.volume_profile.py'),
            names='Monitor.Entropy.VolumeProfile.15'
        )
    )

    tree.extend(
        nodes=FactorNode.from_file(
            file=pathlib.Path(__file__).parents[1].joinpath('_aux', '_aux.py'),
            names=[
                'Monitor.Aux.Adaptive.Index.30',
            ]
        )
    )

    tree.simulate(
        start_date=START_DATE,
        end_date=END_DATE,
        dtype=['TradeData'],
        pred_var=list(str(_) for _ in FutureTopic),
        index_name='000016.SH',
        pred_target='IH_MAIN',
        strategy_mode='sampling',
        # dev_pool=False,
        override_cache=False,
    )


def main():
    profile_cn_override()

    eval_all()
    # eval_individual()
    # sim()

    safe_exit()


if __name__ == '__main__':
    main()
