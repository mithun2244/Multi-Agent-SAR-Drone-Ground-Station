"""One place to seed every global random number generator.

    from ..utils.seed import DEFAULT_SEED, set_global_seed
    set_global_seed(42)

    python -m src.utils.seed        # what it can reach here, and a self-check

What this reaches, and what it does not
---------------------------------------
**Nothing in `src/` draws from a global RNG.** Every random source in this
system is an isolated `random.Random(seed)` — the stub detectors, the Path
Monte-Carlo, the dataset splits, the mock drone feed. That is deliberate: a
fixed validation split has to mean the same thing no matter what else the
process did first, and a run whose results depend on module import order is not
reproducible, only lucky. Seeding `random` globally therefore changes none of
this system's own output.

So this function exists for the libraries we *do not* own — torch and numpy
underneath ultralytics during training, where the global generators are the only
handle there is. It is the belt to the explicit seeds' braces.

Which means a `--seed` flag has to do two things, and every caller here does
both: call this, *and* thread the number into the explicit seed the code already
takes. A flag that only called this would be decorative everywhere except
training.

`PYTHONHASHSEED` is set for completeness, and it only affects processes started
*after* this call — the interpreter fixed its own hash seed before main() ran.
That still matters, because a dataloader spawns workers.
"""

import os
import random
import sys

DEFAULT_SEED = 42


def _seed_numpy(seed):
    import numpy                                    # noqa: PLC0415  (optional)

    numpy.random.seed(seed)
    return True


def _seed_torch(seed):
    import torch                                    # noqa: PLC0415  (optional)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return True


_BACKENDS = (("numpy", _seed_numpy), ("torch", _seed_torch))


def set_global_seed(seed=DEFAULT_SEED):
    """Seed every global RNG reachable in this process.

    Returns the names of what was actually seeded, so a caller can print the
    truth rather than claim determinism it did not get. numpy and torch are
    optional — neither is a dependency of the ground station, and an absent one
    is a normal outcome, not a failure.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)         # child processes only
    seeded = ["random"]

    for name, apply in _BACKENDS:
        try:
            if apply(seed):
                seeded.append(name)
        except ImportError:
            continue                                 # not installed, not a problem
    return tuple(seeded)


def describe(seeded, seed):
    """One line for a script's header."""
    missing = [name for name, _ in _BACKENDS if name not in seeded]
    return (f"seed {seed} — seeded {', '.join(seeded)}"
            + (f"; not installed: {', '.join(missing)}" if missing else ""))


def selfcheck():
    seeded = set_global_seed(1234)
    assert "random" in seeded

    set_global_seed(7)
    first = [random.random() for _ in range(5)]
    set_global_seed(7)
    assert [random.random() for _ in range(5)] == first, "same seed, same sequence"
    set_global_seed(8)
    assert [random.random() for _ in range(5)] != first, "a different seed differs"

    assert os.environ["PYTHONHASHSEED"] == "8"

    # The point of the docstring, asserted: an isolated generator is untouched
    # by whatever the global one is doing, which is why the explicit seeds are
    # what actually make this system reproducible.
    set_global_seed(1)
    isolated = random.Random(99)
    expected = [isolated.random() for _ in range(3)]
    set_global_seed(2)
    random.random()                                  # disturb the global stream
    replay = random.Random(99)
    assert [replay.random() for _ in range(3)] == expected

    try:
        set_global_seed("42")
        raise AssertionError("a non-integer seed must be refused")
    except TypeError:
        pass

    print("  ok  the same seed replays the same global sequence")
    print("  ok  a different seed does not")
    print("  ok  PYTHONHASHSEED is set for child processes")
    print("  ok  an isolated Random(seed) is unaffected by the global seed")
    print("  ok  a non-integer seed is refused")
    print("\n5 checks passed")
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv[1:]:
        sys.exit(selfcheck())
    seed = DEFAULT_SEED
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    print(f"  {describe(set_global_seed(seed), seed)}")
