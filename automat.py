from aiogram.dispatcher.filters.state import State, StatesGroup


class MyState(StatesGroup):
    stop = State()
    waiting_for_image_monet = State()
    waiting_for_image_nst_1 = State()
    waiting_for_image_nst_2 = State()
