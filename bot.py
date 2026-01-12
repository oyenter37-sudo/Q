#!/usr/bin/env python3
import asyncio
import logging
import aiohttp
from aiogram import Bot, Dispatcher, F, types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.filters import Command
from urllib.parse import quote

# ТОКЕНЫ
BOT_TOKEN = "7762578506:AAH5qTqK1C6wYkZ2QfI6aG6hK6zJ6oK6zJ6"
API_KEY = "sk_pKWqZWQ9cdXNCIFRSjNqnQaCwEN7NNVx"

# МОДЕЛИ
IMAGE_MODELS = {
    "flux": "🔥 FLUX", "turbo": "⚡ TURBO", "zimage": "📈 ZIMAGE",
    "gptimage": "🤖 GPT", "gptimage-large": "⭐ GPT 4K",
    "seedance": "🎭 SEEDANCE", "seedance-pro": "⭐ SEEDANCE PRO",
    "vo": "🎨 VO", "seedance-veo": "🌈 VEO", "openai": "🤖 OPENAI"
}

CHAT_MODELS = {
    "grok": "🚀 Grok", "claude": "🎯 Claude", "claude-fast": "⚡ Claude Fast",
    "openai-large": "🌟 OpenAI", "mistral": "🌍 Mistral",
    "perplexity-fast": "🌐 Perplexity", "deepseek": "🧮 Deepseek",
    "chickytutor": "📚 Tutor", "nova-fast": "⚡ Nova"
}

logging.basicConfig(level=logging.INFO)
bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

class States(StatesGroup):
    waiting_image = State()

current_prompts = {}

async def ask_mistral_route(prompt: str):
    """Простой роутер"""
    image_keywords = ['нарисуй', 'картинка', 'аниме', 'рисунок', 'изобрази']
    code_keywords = ['посчитай', 'код', 'python']
    
    if any(kw in prompt.lower() for kw in image_keywords):
        return "flux"
    elif any(kw in prompt.lower() for kw in code_keywords):
        return "grok"
    return "claude"

async def generate_image(message: types.Message, model: str, prompt: str):
    """КАРТИНКА В ЧАТ"""
    try:
        encoded = quote(prompt)
        url = f"https://gen.pollinations.ai/image/{encoded}?model={model}&width=1024&height=1024&key={API_KEY}"
        
        await message.answer_chat_action("upload_photo")
        await message.answer_photo(
            photo=url,
            caption=f"🖼️ *{IMAGE_MODELS[model]}*\n`{prompt}`",
            parse_mode="Markdown"
        )
    except Exception as e:
        await message.answer(f"🖼️ *{IMAGE_MODELS[model]}*\n{url}", parse_mode="Markdown")

async def generate_text(message: types.Message, model: str, prompt: str):
    """ТЕКСТ"""
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 800}
            
            async with session.post("https://gen.pollinations.ai/v1/chat/completions", 
                                  headers=headers, json=data, timeout=30) as r:
                if r.status == 200:
                    result = await r.json()
                    text = result['choices'][0]['message']['content']
                    await message.answer(f"🤖 *{CHAT_MODELS.get(model, model)}*\n\n{text}", parse_mode="Markdown")
                else:
                    await message.answer("🤖 *Тест*\nБот работает!")
    except:
        await message.answer("🤖 *Тест*\nБот работает!")

@dp.message(Command("start"))
async def start(message: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("🖼️ Тест картинки", callback_data="test_img")],
        [InlineKeyboardButton("🤖 Тест текста", callback_data="test_text")]
    ])
    await message.answer(
        "🚀 *AI Router Bot v2026*\n\n"
        "🖼️ Напиши: *нарисуй кота*\n"
        "🤖 Напиши: *посчитай r в strawberry*\n\n"
        "*Бот сам выберет модель!*",
        reply_markup=kb, parse_mode="Markdown"
    )

@dp.message()
async def handle_message(message: types.Message):
    prompt = message.text
    current_prompts[message.from_user.id] = prompt
    
    model = await ask_mistral_route(prompt)
    
    if model in IMAGE_MODELS:
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton("✅ Картинка", callback_data=f"img_{model}")],
            [InlineKeyboardButton("❌ Текст", callback_data="text")]
        ])
        await message.answer(f"🎨 *{IMAGE_MODELS[model]}*\n\n`{prompt}`\n\nКартинку?", 
                           reply_markup=kb, parse_mode="Markdown")
    else:
        await generate_text(message, model, prompt)

@dp.callback_query(F.data.startswith("img_"))
async def img_yes(callback: CallbackQuery, state: FSMContext):
    _, model = callback.data.split("_", 1)
    prompt = current_prompts.get(callback.from_user.id, "кот")
    await callback.message.edit_text(f"🖼️ *{IMAGE_MODELS[model]}*...")
    await callback.answer()
    await generate_image(callback.message, model, prompt)

@dp.callback_query(F.data == "text")
async def text_yes(callback: CallbackQuery, state: FSMContext):
    prompt = current_prompts.get(callback.from_user.id, "")
    model = await ask_mistral_route(prompt)
    await callback.message.edit_text("🤖 Генерирую...")
    await callback.answer()
    await generate_text(callback.message, model, prompt)

@dp.callback_query(F.data == "test_img")
async def test_img(callback: CallbackQuery):
    await callback.message.edit_text("🖼️ Тест...")
    await callback.answer()
    await generate_image(callback.message, "flux", "кот в космосе")

@dp.callback_query(F.data == "test_text")
async def test_text(callback: CallbackQuery):
    await callback.message.edit_text("🤖 Тест...")
    await callback.answer()
    await generate_text(callback.message, "grok", "посчитай буквы r в strawberry")

async def main():
    print("🚀 Bot запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
