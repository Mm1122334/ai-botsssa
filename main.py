import os
import discord
from discord import app_commands
from discord.ext import commands
import google.generativeai as genai
import asyncio
from collections import defaultdict
from threading import Thread
from flask import Flask
import logging
import aiohttp
from PIL import Image
import io
from langdetect import detect, LangDetectException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not DISCORD_TOKEN or not GEMINI_API_KEY:
    raise ValueError("❌ يرجى التأكد من وجود DISCORD_TOKEN و GEMINI_API_KEY في المتغيرات السرية")

genai.configure(api_key=GEMINI_API_KEY)

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

model = genai.GenerativeModel(
    'gemini-2.0-flash-exp',
    safety_settings=safety_settings
)

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

conversation_history = defaultdict(list)
auto_reply_channels = set()
user_last_language = defaultdict(lambda: 'ar')

MAX_HISTORY = 10
MAX_MESSAGE_LENGTH = 2000

app = Flask(__name__)

@app.route('/')
def health_check():
    return {'status': 'ok', 'bot': 'running'}, 200

@app.route('/health')
def health():
    return {'status': 'healthy'}, 200

def run_flask():
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

def split_message(text, max_length=MAX_MESSAGE_LENGTH):
    messages = []
    while len(text) > max_length:
        split_pos = text.rfind('\n', 0, max_length)
        if split_pos == -1:
            split_pos = text.rfind(' ', 0, max_length)
        if split_pos == -1:
            split_pos = max_length
        
        messages.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    
    if text:
        messages.append(text)
    
    return messages

def detect_language(text):
    if not text:
        return 'unknown'
    
    text = text.strip()
    
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin_chars = sum(1 for c in text if c.isalpha() and not ('\u0600' <= c <= '\u06FF'))
    total_chars = arabic_chars + latin_chars
    
    if total_chars == 0:
        return 'unknown'
    
    if arabic_chars > 0:
        return 'ar'
    elif latin_chars > 0:
        return 'en'
    else:
        return 'unknown'

def check_name_question(text):
    name_patterns = [
        'وش اسمك', 'ماهو اسمك', 'شو اسمك', 'شنو اسمك',
        'ما اسمك', 'ايش اسمك', 'وين اسمك',
        'what is your name', 'whats your name', "what's your name",
        'who are you', 'your name'
    ]
    text_lower = text.lower().strip()
    for pattern in name_patterns:
        if pattern in text_lower:
            return True
    return False

def detect_image_edit_request(text):
    if not text:
        return None, None
    
    text_lower = text.lower().strip()
    
    rotate_patterns = [
        ('دور', 'rotate'), ('دوره', 'rotate'), ('دورها', 'rotate'),
        ('لف', 'rotate'), ('لفه', 'rotate'), ('لفها', 'rotate'),
        ('rotate', 'rotate'), ('turn', 'rotate')
    ]
    
    for ar, en in rotate_patterns:
        if ar in text_lower:
            if '90' in text_lower:
                return 'rotate', 90
            elif '180' in text_lower:
                return 'rotate', 180
            elif '270' in text_lower:
                return 'rotate', 270
            else:
                return 'rotate', 90
    
    filter_patterns = {
        'أبيض وأسود': 'grayscale',
        'ابيض واسود': 'grayscale',
        'black and white': 'grayscale',
        'grayscale': 'grayscale',
        'رمادي': 'grayscale',
        
        'سيبيا': 'sepia',
        'sepia': 'sepia',
        'قديم': 'sepia',
        
        'ضبابي': 'blur',
        'blur': 'blur',
        'تضبيب': 'blur',
        
        'حاد': 'sharpen',
        'sharpen': 'sharpen',
        'واضح': 'sharpen',
        
        'ساطع': 'bright',
        'bright': 'bright',
        'فاتح': 'bright',
        'أفتح': 'bright',
        
        'تباين': 'contrast',
        'contrast': 'contrast'
    }
    
    for pattern, filter_type in filter_patterns.items():
        if pattern in text_lower:
            return 'filter', filter_type
    
    return None, None

async def process_image_edit(image_url, edit_type, edit_param):
    try:
        img = await download_image(image_url)
        if not img:
            return None, "❌ فشل تحميل الصورة!"
        
        if edit_type == 'rotate':
            edited = await asyncio.to_thread(rotate_image, img, edit_param)
            filename = f"rotated_{edit_param}.png"
            message = f"✅ تم تدوير الصورة {edit_param} درجة!"
        elif edit_type == 'filter':
            edited = await asyncio.to_thread(apply_filter, img, edit_param)
            filename = f"filtered_{edit_param}.png"
            filter_names = {
                'grayscale': 'أبيض وأسود',
                'sepia': 'سيبيا',
                'blur': 'ضبابي',
                'sharpen': 'حاد',
                'bright': 'ساطع',
                'contrast': 'تباين عالي'
            }
            message = f"✅ تم تطبيق فلتر {filter_names.get(edit_param, edit_param)}!"
        else:
            return None, None
        
        img_bytes = await asyncio.to_thread(image_to_bytes, edited)
        return discord.File(fp=img_bytes, filename=filename), message
        
    except Exception as e:
        logger.error(f"خطأ في معالجة تعديل الصورة: {e}")
        return None, f"❌ حدث خطأ في تعديل الصورة: {str(e)}"

async def download_image(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    return Image.open(io.BytesIO(image_data))
        return None
    except Exception as e:
        logger.error(f"خطأ في تحميل الصورة: {e}")
        return None

def rotate_image(image, degrees):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return image.rotate(degrees, expand=True)

def resize_image(image, width, height):
    return image.resize((width, height), Image.Resampling.LANCZOS)

def apply_filter(image, filter_type):
    from PIL import ImageFilter, ImageEnhance, ImageOps
    
    if filter_type == "blur":
        return image.filter(ImageFilter.BLUR)
    elif filter_type == "sharpen":
        return image.filter(ImageFilter.SHARPEN)
    elif filter_type == "grayscale":
        return ImageOps.grayscale(image).convert("RGB")
    elif filter_type == "sepia":
        grayscale = ImageOps.grayscale(image)
        sepia = Image.new("RGB", image.size)
        pixels = sepia.load()
        gray_pixels = grayscale.load()
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                gray = gray_pixels[i, j]
                pixels[i, j] = (int(gray * 1.0), int(gray * 0.95), int(gray * 0.82))
        return sepia
    elif filter_type == "bright":
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.5)
    elif filter_type == "contrast":
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.5)
    else:
        return image

def crop_image(image, left, top, right, bottom):
    return image.crop((left, top, right, bottom))

def add_text_to_image(image, text, position=(10, 10), color=(255, 255, 255)):
    from PIL import ImageDraw, ImageFont
    
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    draw.text(position, text, fill=color, font=font)
    return image

def image_to_bytes(image):
    img_byte_arr = io.BytesIO()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def _generate_content_sync(content):
    return model.generate_content(content)

async def get_ai_response(user_id, prompt, image_urls=None):
    try:
        detected_lang = detect_language(prompt)
        if detected_lang == 'unknown':
            detected_lang = user_last_language[user_id]
        else:
            user_last_language[user_id] = detected_lang
        
        if check_name_question(prompt):
            if detected_lang == 'ar':
                custom_response = "انا ذكاء اصطناعي متطور بواسطة سيرفر\nhaven H-V"
            else:
                custom_response = "I am an advanced AI developed by haven H-V server"
            conversation_history[user_id].append({
                'user': prompt,
                'assistant': custom_response,
                'language': detected_lang
            })
            if len(conversation_history[user_id]) > MAX_HISTORY:
                conversation_history[user_id].pop(0)
            return custom_response
        
        history = conversation_history[user_id]
        
        full_context = ""
        for entry in history:
            full_context += f"{entry['user']}\n{entry['assistant']}\n\n"
        full_context += prompt
        
        if image_urls and len(image_urls) > 0:
            images = []
            for url in image_urls:
                img = await download_image(url)
                if img:
                    images.append(img)
            
            if images:
                content = [full_context] + images
                response = await asyncio.to_thread(_generate_content_sync, content)
            else:
                response = await asyncio.to_thread(_generate_content_sync, full_context)
        else:
            response = await asyncio.to_thread(_generate_content_sync, full_context)
        
        ai_response = response.text
        
        conversation_history[user_id].append({
            'user': prompt,
            'assistant': ai_response,
            'language': detected_lang
        })
        
        if len(conversation_history[user_id]) > MAX_HISTORY:
            conversation_history[user_id].pop(0)
        
        return ai_response
    
    except Exception as e:
        error_str = str(e)
        logger.error(f"خطأ في الحصول على رد من Gemini: {e}")
        
        if '429' in error_str or 'Resource exhausted' in error_str:
            return "⏳ API quota exceeded. Please wait a few minutes and try again.\nتم تجاوز حد الاستخدام. يرجى الانتظار بضع دقائق والمحاولة مرة أخرى."
        elif 'SAFETY' in error_str.upper() or 'blocked' in error_str.lower():
            return "⚠️ Sorry, I cannot answer this question for safety reasons.\nعذراً، لا يمكنني الإجابة على هذا السؤال لأسباب السلامة."
        else:
            return f"❌ An error occurred. Please try again.\nحدث خطأ. يرجى المحاولة مرة أخرى."

async def send_long_message(channel, text):
    messages = split_message(text)
    for i, msg in enumerate(messages):
        await channel.send(msg)
        if i < len(messages) - 1:
            await asyncio.sleep(0.5)

@bot.event
async def on_ready():
    logger.info(f'✅ تم تسجيل الدخول كـ {bot.user}')
    try:
        synced = await bot.tree.sync()
        logger.info(f'✅ تم مزامنة {len(synced)} أمر')
    except Exception as e:
        logger.error(f'❌ خطأ في المزامنة: {e}')
    
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="/help للمساعدة"
        )
    )

@bot.tree.command(name="ask", description="اسأل البوت أي سؤال")
@app_commands.describe(question="السؤال الذي تريد طرحه")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    
    try:
        response = await get_ai_response(interaction.user.id, question)
        messages = split_message(response)
        
        await interaction.followup.send(messages[0])
        
        for msg in messages[1:]:
            await interaction.channel.send(msg)
            await asyncio.sleep(0.5)
    
    except Exception as e:
        logger.error(f"خطأ في أمر ask: {e}")
        await interaction.followup.send(f"❌ حدث خطأ: {str(e)}")

@bot.tree.command(name="help", description="عرض دليل استخدام البوت")
async def help_command(interaction: discord.Interaction):
    help_text = """
# 📚 دليل استخدام البوت

## 🎯 الأوامر الأساسية:
**`/ask [سؤال]`** - اسأل البوت أي سؤال
**`/help`** - عرض هذا الدليل
**`/clear`** - مسح سجل محادثتك
**`/ping`** - فحص سرعة استجابة البوت

## 🎨 أوامر تعديل الصور:
**`/rotate [صورة] [درجات]`** - تدوير الصورة (90, 180, 270)
**`/resize [صورة] [عرض] [ارتفاع]`** - تغيير حجم الصورة
**`/filter [صورة] [نوع]`** - تطبيق فلتر (blur, sharpen, grayscale, sepia, bright, contrast)
**`/crop [صورة] [left] [top] [right] [bottom]`** - قص الصورة
**`/addtext [صورة] [نص] [x] [y]`** - إضافة نص على الصورة

## 🛡️ أوامر الإدارة (للمشرفين فقط):
**`/setchannel`** - تفعيل الرد التلقائي في هذه القناة
**`/removechannel`** - إلغاء الرد التلقائي من هذه القناة
**`/listchannels`** - عرض القنوات المفعلة
**`/clearallchannels`** - إزالة جميع القنوات المفعلة

## 💬 طرق التفاعل:
✅ **Slash Commands** - استخدم الأوامر أعلاه
✅ **@منشن** - اذكر البوت في رسالتك (@{})
✅ **الرد** - رد على رسالة البوت مباشرة
✅ **رد تلقائي** - في القنوات المفعلة سيرد البوت تلقائياً

## 🧠 إدارة السياق:
• يحفظ البوت آخر {} تبادلات لكل مستخدم
• يفهم الأسئلة بناءً على السياق السابق
• استخدم `/clear` لبدء محادثة جديدة

## 🤖 مدعوم بـ:
**Google Gemini 2.0 Flash Experimental**
نموذج ذكاء اصطناعي متقدم للإجابة على أسئلتك

---
💡 **نصيحة**: جرب سؤال البوت عن أي موضوع أو تعديل صورك!
""".format(bot.user.mention, MAX_HISTORY)
    
    await interaction.response.send_message(help_text)

@bot.tree.command(name="clear", description="مسح سجل محادثتك مع البوت")
async def clear(interaction: discord.Interaction):
    user_id = interaction.user.id
    if user_id in conversation_history:
        conversation_history[user_id].clear()
        await interaction.response.send_message("✅ تم مسح سجل محادثتك بنجاح!")
    else:
        await interaction.response.send_message("ℹ️ لا يوجد سجل محادثات لمسحه.")

@bot.tree.command(name="ping", description="فحص سرعة استجابة البوت")
async def ping(interaction: discord.Interaction):
    latency = round(bot.latency * 1000)
    await interaction.response.send_message(
        f"🏓 بونج!\n⚡ سرعة الاستجابة: `{latency}ms`"
    )

@bot.tree.command(name="setchannel", description="تفعيل الرد التلقائي في هذه القناة (للمشرفين)")
async def setchannel(interaction: discord.Interaction):
    if not interaction.user.guild_permissions.manage_channels:
        await interaction.response.send_message("❌ يجب أن تكون مشرفاً لاستخدام هذا الأمر!")
        return
    
    channel_id = interaction.channel_id
    auto_reply_channels.add(channel_id)
    await interaction.response.send_message(
        f"✅ تم تفعيل الرد التلقائي في القناة <#{channel_id}>"
    )

@bot.tree.command(name="removechannel", description="إلغاء الرد التلقائي من هذه القناة (للمشرفين)")
async def removechannel(interaction: discord.Interaction):
    if not interaction.user.guild_permissions.manage_channels:
        await interaction.response.send_message("❌ يجب أن تكون مشرفاً لاستخدام هذا الأمر!")
        return
    
    channel_id = interaction.channel_id
    if channel_id in auto_reply_channels:
        auto_reply_channels.remove(channel_id)
        await interaction.response.send_message(
            f"✅ تم إلغاء الرد التلقائي من القناة <#{channel_id}>"
        )
    else:
        await interaction.response.send_message("ℹ️ هذه القناة غير مفعلة للرد التلقائي.")

@bot.tree.command(name="listchannels", description="عرض القنوات المفعلة للرد التلقائي (للمشرفين)")
async def listchannels(interaction: discord.Interaction):
    if not interaction.user.guild_permissions.manage_channels:
        await interaction.response.send_message("❌ يجب أن تكون مشرفاً لاستخدام هذا الأمر!")
        return
    
    if not auto_reply_channels:
        await interaction.response.send_message("ℹ️ لا توجد قنوات مفعلة للرد التلقائي.")
        return
    
    channels_list = "\n".join([f"• <#{ch_id}>" for ch_id in auto_reply_channels])
    await interaction.response.send_message(
        f"📋 **القنوات المفعلة للرد التلقائي:**\n{channels_list}"
    )

@bot.tree.command(name="clearallchannels", description="إزالة جميع القنوات المفعلة (للمشرفين)")
async def clearallchannels(interaction: discord.Interaction):
    if not interaction.user.guild_permissions.manage_channels:
        await interaction.response.send_message("❌ يجب أن تكون مشرفاً لاستخدام هذا الأمر!")
        return
    
    count = len(auto_reply_channels)
    auto_reply_channels.clear()
    await interaction.response.send_message(
        f"✅ تم إزالة {count} قناة من الرد التلقائي."
    )

@bot.tree.command(name="rotate", description="تدوير الصورة")
@app_commands.describe(
    image="الصورة المراد تدويرها",
    degrees="درجة التدوير"
)
@app_commands.choices(degrees=[
    app_commands.Choice(name="90 درجة", value=90),
    app_commands.Choice(name="180 درجة", value=180),
    app_commands.Choice(name="270 درجة", value=270),
])
async def rotate(interaction: discord.Interaction, image: discord.Attachment, degrees: app_commands.Choice[int]):
    await interaction.response.defer(thinking=True)
    
    try:
        if not image.content_type or not image.content_type.startswith('image/'):
            await interaction.followup.send("❌ يجب أن يكون المرفق صورة!")
            return
        
        img = await download_image(image.url)
        if not img:
            await interaction.followup.send("❌ فشل تحميل الصورة!")
            return
        
        rotated = await asyncio.to_thread(rotate_image, img, degrees.value)
        img_bytes = await asyncio.to_thread(image_to_bytes, rotated)
        
        await interaction.followup.send(
            f"✅ تم تدوير الصورة {degrees.value} درجة!",
            file=discord.File(fp=img_bytes, filename=f"rotated_{degrees.value}.png")
        )
    except Exception as e:
        logger.error(f"خطأ في تدوير الصورة: {e}")
        await interaction.followup.send(f"❌ حدث خطأ: {str(e)}")

@bot.tree.command(name="resize", description="تغيير حجم الصورة")
@app_commands.describe(
    image="الصورة المراد تغيير حجمها",
    width="العرض الجديد بالبكسل",
    height="الارتفاع الجديد بالبكسل"
)
async def resize(interaction: discord.Interaction, image: discord.Attachment, width: int, height: int):
    await interaction.response.defer(thinking=True)
    
    try:
        if not image.content_type or not image.content_type.startswith('image/'):
            await interaction.followup.send("❌ يجب أن يكون المرفق صورة!")
            return
        
        if width <= 0 or height <= 0 or width > 4000 or height > 4000:
            await interaction.followup.send("❌ الأبعاد يجب أن تكون بين 1 و 4000 بكسل!")
            return
        
        img = await download_image(image.url)
        if not img:
            await interaction.followup.send("❌ فشل تحميل الصورة!")
            return
        
        resized = await asyncio.to_thread(resize_image, img, width, height)
        img_bytes = await asyncio.to_thread(image_to_bytes, resized)
        
        await interaction.followup.send(
            f"✅ تم تغيير حجم الصورة إلى {width}x{height}!",
            file=discord.File(fp=img_bytes, filename=f"resized_{width}x{height}.png")
        )
    except Exception as e:
        logger.error(f"خطأ في تغيير حجم الصورة: {e}")
        await interaction.followup.send(f"❌ حدث خطأ: {str(e)}")

@bot.tree.command(name="filter", description="تطبيق فلتر على الصورة")
@app_commands.describe(
    image="الصورة المراد تطبيق الفلتر عليها",
    filter_type="نوع الفلتر"
)
@app_commands.choices(filter_type=[
    app_commands.Choice(name="🌫️ ضبابي (Blur)", value="blur"),
    app_commands.Choice(name="✨ حاد (Sharpen)", value="sharpen"),
    app_commands.Choice(name="⚫ أبيض وأسود (Grayscale)", value="grayscale"),
    app_commands.Choice(name="🟤 سيبيا (Sepia)", value="sepia"),
    app_commands.Choice(name="☀️ ساطع (Bright)", value="bright"),
    app_commands.Choice(name="🎨 تباين عالي (Contrast)", value="contrast"),
])
async def filter_cmd(interaction: discord.Interaction, image: discord.Attachment, filter_type: app_commands.Choice[str]):
    await interaction.response.defer(thinking=True)
    
    try:
        if not image.content_type or not image.content_type.startswith('image/'):
            await interaction.followup.send("❌ يجب أن يكون المرفق صورة!")
            return
        
        img = await download_image(image.url)
        if not img:
            await interaction.followup.send("❌ فشل تحميل الصورة!")
            return
        
        filtered = await asyncio.to_thread(apply_filter, img, filter_type.value)
        img_bytes = await asyncio.to_thread(image_to_bytes, filtered)
        
        await interaction.followup.send(
            f"✅ تم تطبيق فلتر {filter_type.name}!",
            file=discord.File(fp=img_bytes, filename=f"filtered_{filter_type.value}.png")
        )
    except Exception as e:
        logger.error(f"خطأ في تطبيق الفلتر: {e}")
        await interaction.followup.send(f"❌ حدث خطأ: {str(e)}")

@bot.tree.command(name="crop", description="قص الصورة")
@app_commands.describe(
    image="الصورة المراد قصها",
    left="الحافة اليسرى",
    top="الحافة العلوية",
    right="الحافة اليمنى",
    bottom="الحافة السفلية"
)
async def crop(interaction: discord.Interaction, image: discord.Attachment, left: int, top: int, right: int, bottom: int):
    await interaction.response.defer(thinking=True)
    
    try:
        if not image.content_type or not image.content_type.startswith('image/'):
            await interaction.followup.send("❌ يجب أن يكون المرفق صورة!")
            return
        
        img = await download_image(image.url)
        if not img:
            await interaction.followup.send("❌ فشل تحميل الصورة!")
            return
        
        if left < 0 or top < 0 or right > img.width or bottom > img.height or left >= right or top >= bottom:
            await interaction.followup.send(f"❌ إحداثيات القص غير صحيحة! أبعاد الصورة: {img.width}x{img.height}")
            return
        
        cropped = await asyncio.to_thread(crop_image, img, left, top, right, bottom)
        img_bytes = await asyncio.to_thread(image_to_bytes, cropped)
        
        await interaction.followup.send(
            f"✅ تم قص الصورة!",
            file=discord.File(fp=img_bytes, filename=f"cropped.png")
        )
    except Exception as e:
        logger.error(f"خطأ في قص الصورة: {e}")
        await interaction.followup.send(f"❌ حدث خطأ: {str(e)}")

@bot.tree.command(name="addtext", description="إضافة نص على الصورة")
@app_commands.describe(
    image="الصورة المراد إضافة النص عليها",
    text="النص المراد إضافته",
    x="موضع X للنص (اختياري)",
    y="موضع Y للنص (اختياري)"
)
async def addtext(interaction: discord.Interaction, image: discord.Attachment, text: str, x: int = 10, y: int = 10):
    await interaction.response.defer(thinking=True)
    
    try:
        if not image.content_type or not image.content_type.startswith('image/'):
            await interaction.followup.send("❌ يجب أن يكون المرفق صورة!")
            return
        
        img = await download_image(image.url)
        if not img:
            await interaction.followup.send("❌ فشل تحميل الصورة!")
            return
        
        with_text = await asyncio.to_thread(add_text_to_image, img.copy(), text, (x, y))
        img_bytes = await asyncio.to_thread(image_to_bytes, with_text)
        
        await interaction.followup.send(
            f"✅ تم إضافة النص على الصورة!",
            file=discord.File(fp=img_bytes, filename=f"with_text.png")
        )
    except Exception as e:
        logger.error(f"خطأ في إضافة النص: {e}")
        await interaction.followup.send(f"❌ حدث خطأ: {str(e)}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    should_reply = False
    
    if bot.user in message.mentions:
        should_reply = True
    
    elif message.reference and message.reference.resolved:
        if message.reference.resolved.author == bot.user:
            should_reply = True
    
    elif message.channel.id in auto_reply_channels:
        should_reply = True
    
    if should_reply:
        async with message.channel.typing():
            content = message.content.replace(f'<@{bot.user.id}>', '').strip()
            
            image_urls = []
            if message.attachments:
                for attachment in message.attachments:
                    if attachment.content_type and attachment.content_type.startswith('image/'):
                        image_urls.append(attachment.url)
            
            if not content and not image_urls:
                await message.reply("👋 Hi! How can I help you?\nمرحباً! كيف يمكنني مساعدتك؟")
                return
            
            if image_urls and content:
                edit_type, edit_param = detect_image_edit_request(content)
                if edit_type:
                    try:
                        file, edit_message = await process_image_edit(image_urls[0], edit_type, edit_param)
                        if file:
                            await message.reply(edit_message, file=file)
                        else:
                            await message.reply(edit_message)
                        return
                    except Exception as e:
                        logger.error(f"خطأ في تعديل الصورة التلقائي: {e}")
            
            if not content and image_urls:
                content = "What's in this image?"
            
            try:
                response = await get_ai_response(message.author.id, content, image_urls)
                await send_long_message(message.channel, response)
            except Exception as e:
                logger.error(f"خطأ في معالجة الرسالة: {e}")
                await message.reply(f"❌ حدث خطأ: {str(e)}")
    
    await bot.process_commands(message)

def main():
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    logger.info("🚀 جاري تشغيل البوت...")
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()
