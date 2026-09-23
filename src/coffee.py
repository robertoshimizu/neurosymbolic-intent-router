from statemachine import StateChart, State


class CoffeeOrder(StateChart):
    # Define the states
    pending = State(initial=True)
    preparing = State()
    ready = State()
    picked_up = State(final=True)

    # Define events — each one groups one or more transitions
    start = pending.to(preparing)
    finish = preparing.to(ready)
    pick_up = ready.to(picked_up)


if __name__ == "__main__":
    order = CoffeeOrder()
    print("pending active:", order.pending.is_active)

    order.send("start")
    print("preparing active:", order.preparing.is_active)

    order.send("finish")
    order.send("pick_up")
    print("picked_up active:", order.picked_up.is_active)

    # Events can also be called as methods
    order = CoffeeOrder()
    order.start()
    print("preparing active (via method):", order.preparing.is_active)
