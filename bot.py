import os
import aiohttp

from aiogram.dispatcher import FSMContext
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from PIL import Image
import time

from config import TOKEN, write_start, write_help
from user import user
from automat import MyState
from utils import gen_idx
from models import Model_monet
from nst import nst_model

proxy_host = 'socks5://178.128.203.1:1080'
PROXY_AUTH = aiohttp.BasicAuth(login='student', password='TH8FwlMMwWvbJF8FYcq0')
bot = Bot(token=TOKEN,
          proxy=proxy_host,
          proxy_auth=PROXY_AUTH)

storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
users = {}

model_monet = Model_monet()
#model_nst = nst_model()
gen_idx_ = gen_idx()


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    print(message)
    await message.answer(write_start)


@dp.message_handler(commands=['help'])
async def help(message: types.Message):
    await message.answer(write_help)


@dp.message_handler(commands=['style_transfer_monet'], state='*')
async def style_transfer(message: types.Message):
    await MyState.waiting_for_image_monet.set()
    await message.answer('Пришлите картинку, которую необходимо стилизировать')


@dp.message_handler(state=MyState.waiting_for_image_monet, content_types=types.ContentTypes.PHOTO)
async def style_transfer_step_2(message: types.Message, state: FSMContext):
    await message.answer("Отлично, обработка фото началась!!!")
    await MyState.stop.set()
    async with state.proxy() as proxy:
        proxy.setdefault('image_id', message.photo)
        proxy.clear()

    idx = gen_idx_.__next__()
    await message.photo[-1].download('images/image_{}_{}.jpg'.format(message['from']['id'], idx))

    await model_monet.predict('images/image_{}_{}'.format(message['from']['id'], idx))

    await bot.send_photo(message.from_user.id,
                         types.input_file.InputFile('images/image_{}_{}_pre.jpg'.format(message['from']['id'], idx)))

'''
@dp.message_handler(commands=['style_transfer_nst'], state='*')
async def nst(message: types.Message):
    await MyState.waiting_for_image_nst_1.set()
    await message.answer('Пришлите картинку - контент')


@dp.message_handler(state=MyState.waiting_for_image_nst_1, content_types=types.ContentTypes.PHOTO)
async def nst_step_2(message: types.Message, state: FSMContext):
    await message.answer('Пришлите картинку - стиль')
    await MyState.waiting_for_image_nst_2.set()
    async with state.proxy() as proxy:
        proxy.setdefault('image_content_id', message.photo)


@dp.message_handler(state=MyState.waiting_for_image_nst_2, content_types=types.ContentTypes.PHOTO)
async def nst_step_3(message: types.Message, state: FSMContext):
    await message.answer('Отлично! Обработка фото началась!!!')

    await MyState.stop.set()

    async with state.proxy() as proxy:
        content_img = proxy['image_content_id'][-1]
        proxy.clear()
    style_img = message.photo[-1]

    idx = gen_idx_.__next__()
    await content_img.download('images/image_{}_{}_content_.jpg'.format(message['from']['id'], idx))
    await style_img.download('images/image_{}_{}_style_.jpg'.format(message['from']['id'], idx))

    await model_nst.predict('images/image_{}_{}'.format(message['from']['id'], idx))

    await bot.send_photo(message.from_user.id,
                         types.input_file.InputFile('images/image_{}_{}_pre.jpg'.format(message['from']['id'], idx)))
'''

@dp.message_handler(commands=['cancel'], state='*')
async def cancel(message: types.Message, state: FSMContext):
    await MyState.stop.set()

    async with state.proxy() as proxy:
        proxy.clear()

    await message.answer('Отменено')


@dp.message_handler(content_types=types.ContentTypes.PHOTO, state='*')
async def non(message: types.Message):
    await message.answer('Если вы хотите стилизовать изображение, то напишите мне одну из этих команд: \n' + write_help)

if __name__ == '__main__':
    executor.start_polling(dp)
    print('Start!!!')
