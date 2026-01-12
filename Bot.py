#!/usr/bin/env python3
import asyncio
import logging
import aiohttp
from aiogram import Bot, Dispatcher, F, types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import Command
from urllib.parse import quote

# ТОКЕНЫ В КОДЕ
BOT_TOKEN = "7762578506:AAH5qTqK1C6wYkZ2QfI6aG6hK6zJ6oK6zJ6"  # ТВОЙ ТОКЕН
API_KEY = "sk_pKWqZWQ9cdXNCIFRSjNqnQaCwEN7NNVx"

# 🖼️ 10 КАРТИНОК
IMAGE_MODELS = {
    "flux": "🔥 FLUX", "turbo": "⚡ TURBO", "zimage": "📈 ZIMAGE",
    "gptimage": "🤖 GPT", "gptimage-large": "⭐ GPT 4K",
    "seedance": "🎭 SEEDANCE", "seedance-pro": "⭐ SEEDANCE PRO",
    "vo": "🎨 VO", "seedance-veo": "🌈 VEO", "openai": "🤖 OPENAI"
}

# 🤖 9 ЧАТ
CHAT_MODELS = {
    "grok": "🚀 Grok", "claude": "🎯 Claude", "claude-fast": "⚡ Claude Fast",
    "openai-large": "🌟 OpenAI", "mistral": "🌍 Mistral",
    "perplexity-fast": "🌐 Perplexity", "deepseek": "🧮 Deepseek",
    "chickytutor": "📚 Tutor", "nova-fast": "⚡ Nova"
}

MISTRAL_SYSTEM_PROMPT = """
Выбирай модель для промпта.

КАРТИНКА: flux/turbo/seedance (нарисуй, картинка, аниме)
КОД: grok (посчитай, python)
ЛОГИКА: claude
РУССКИЙ: mistral
ФАКТЫ: perplexity-fast

ТОЛЬКО одно слово: flux, grok, claude, turbo, seedance
"""

logging.basicConfig(level=logging.INFO)
bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

class States(StatesGroup):
    waiting_image = State()

user_settings = {}
current_prompts = {}  # Храним промпты для callback

async def ask_mistral_route(prompt: str, no_image=False):
    """Mistral Router"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "mistral",
        "messages": [{"role": "system", "content": MISTRAL_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "max_tokens": 20
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://gen.pollinations.ai/v1/chat/completions", 
                                  headers=headers, json=data, timeout=10) as r:
                if r.status == 200:
                    result = await r.json()
                    model = result['choices'][0]['message']['content'].strip().lower()
                    all_models = {**IMAGE_MODELS, **CHAT_MODELS}
                    if model in all_models:
                        return model
    except:
        pass
    
    # Fallback
    if any(kw in prompt.lower() for kw in ['нарисуй', 'картинка', 'аниме', 'рисунок']):
        return "flux"
    elif any(kw in prompt.lower() for kw in ['посчитай', 'код', 'python']):
        return "grok"
    return "claude"

async def generate_image(message: types.Message, model: str, prompt: str):
    """ОТПРАВЛЯЕТ КАРТИНКУ ПРЯМО В ЧАТ"""
    encoded = quote(prompt)
    url = f"https://gen.pollinations.ai/image/{encoded}?model={model}&width=1024&height=1024&key={API_KEY}"
    
    await message.answer_chat_action("upload_photo")
    
    try:
        await message.answer_photo(
            photo=url,
            caption=f"🖼️ *{IMAGE_MODELS[model]}*\n`{prompt}`",
            parse_mode="Markdown"
        )
    except:
        # Если не грузится как фото - отправляем как документ
        await message.answer_photo(
            photo=url,
            caption=f"🖼️ *{IMAGE_MODELS[model]}*\n`{prompt}`",
            parse_mode="Markdown"
        )

async def generate_text(message: types.Message, model: str, prompt: str):
    """Текст"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 800}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://gen.pollinations.ai/v1/chat/completions", 
                                  headers=headers, json=data, timeout=30) as r:
                if r.status == 200:
                    result = await r.json()
                    text = result['choices'][0]['message']['content']
                    model_name = CHAT_MODELS.get(model, model.upper())
                    await message.answer(f"🤖 *{model_name}*\n\n{text}", parse_mode="Markdown")
                else:
                    await message.answer(f"❌ Ошибка {model}")
    except:
        await message.answer("⏰ Таймаут, попробуй еще раз")

@dp.message(Command("start"))
async def start(message: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("🖼️ Тест картинки", callback_data="test_img")],
        [InlineKeyboardButton("🤖 Тест текста", callback_data="test_text")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")]
    ])
    await message.answer(
        "🚀 *Pollinations AI Router v2026*\n\n"
        "🖼️ 10 моделей картинок\n"
        "🤖 9 моделей чата\n\n"
        "*Напиши любой промпт — бот сам выберет модель!*",
        reply_markup=kb, parse_mode="Markdown"
    )

@dp.message()
async def handle_message(message: types.Message, state: FSMContext):
    prompt = message.text
    user_id = message.from_user.id
    
    # Сохраняем промпт
    current_prompts[user_id] = prompt
    
    model = await ask_mistral_route(prompt)
    
    if model in IMAGE_MODELS:
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton("✅ Картинка", callback_data=f"img_yes_{model}")],
            [InlineKeyboardButton("❌ Текст", callback_data="text_yes")],
            [InlineKeyboardButton("🎨 Другая модель", callback_data="img_choose")]
        ])
        
        await message.answer(
            f"🎨 *{IMAGE_MODELS[model]}*\n\n"
            f"Хотите *картинку*?\n\n"
            f"`{prompt}`",
            reply_markup=kb, parse_mode="Markdown"
        )
        await state.set_state(States.waiting_image)
    else:
        await generate_text(message, model, prompt)

@dp.callback_query(F.data.startswith("img_yes_"))
async def img_yes(callback: CallbackQuery, state: FSMContext):
    _, model = callback.data.split("_", 2)
    user_id = callback.from_user.id
    prompt = current_prompts.get(user_id, "кот")
    
    await callback.message.edit_text(f"🖼️ *{IMAGE_MODELS[model]}* генерирует...")
    await generate_image(callback.message, model, prompt)
    await state.clear()

@dp.callback_query(F.data == "text_yes")
async def text_yes(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    prompt = current_prompts.get(user_id, "")
    model = await ask_mistral_route(prompt, no_image=True)
    
    await callback.message.edit_text("🤖 Генерирую текст...")
    await generate_text(callback.message, model, prompt)
    await state.clear()

@dp.callback_query(F.data == "img_choose")
async def img_choose(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    prompt = current_prompts.get(user_id, "")
    
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(IMAGE_MODELS[k], callback_data=f"img_model_{k}") for k in list(IMAGE_MODELS)[:2]],
        [InlineKeyboardButton(IMAGE_MODELS[k], callback_data=f"img_model_{k}") for k in list(IMAGE_MODELS)[2:4]],
        [InlineKeyboardButton(IMAGE_MODELS[k], callback_data=f"img_model_{k}") for k in list(IMAGE_MODELS)[4:6]],
        [InlineKeyboardButton("🔙 Назад", callback_data="back_main")]
    ])
    
    await callback.message.edit_text(
        f"🎨 *ВЫБЕРИ МОДЕЛЬ ДЛЯ:* `{prompt}`",
        reply_markup=kb, parse_mode="Markdown"
    )

@dp.callback_query(F.data.startswith("img_model_"))
async def img_model_select(callback: CallbackQuery, state: FSMContext):
    model = callback.data.split("img_model_")[1]
    user_id = callback.from_user.id
    prompt = current_prompts.get(user_id, "кот")
    
    await callback.message.edit_text(f"🖼️ *{IMAGE_MODELS[model]}*...")
    await generate_image(callback.message, model, prompt)
    await state.clear()

@dp.callback_query(F.data == "test_img")
async def test_img(callback: CallbackQuery):
    await callback.message.edit_text("🖼️ Тест flux...")
    await generate_image(callback.message, "flux", "кот в космосе")

@dp.callback_query(F.data == "test_text")
async def test_text(callback: CallbackQuery):
    await callback.message.edit_text("🤖 Тест grok...")
    await generate_text(callback.message, "grok", "посчитай буквы r в strawberry")

async def main():
    print("🚀 Pollinations AI Router Bot v2026 запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
