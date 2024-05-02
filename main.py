import google.generativeai as genai

GOOGLE_API_KEY = 'REPLACE YOUR GOOGLE AI API KEY'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.0-pro-latest')

response = model.generate_content(input('Ask Gemini: '))
print(response)