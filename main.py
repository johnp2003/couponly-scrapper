import asyncio
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Set
from supabase import create_client, Client
from playwright.async_api import async_playwright
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()


class CouponScraper:
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.shop_results = {}
        self.processed_shops = set()
        self.max_shops = 2  # Testing with 10 shops only

        # Initialize Supabase client
        if self.supabase_url and self.supabase_key:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        else:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be provided in environment variables")

        # Configure Gemini
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

    def clean_title_text(self, title_text: str) -> str:
        """Clean title text by removing extra whitespace and newlines"""
        if not title_text:
            return 'No title'

        # Step 1: Replace all Unicode whitespace characters with regular spaces
        # This includes regular spaces, newlines, tabs, non-breaking spaces, etc.
        cleaned = re.sub(r'\s+', ' ', title_text, flags=re.UNICODE)

        # Step 2: Handle any remaining non-breaking spaces or special characters
        cleaned = cleaned.replace('\u00A0', ' ')  # Non-breaking space
        cleaned = cleaned.replace('\u2009', ' ')  # Thin space
        cleaned = cleaned.replace('\u200B', '')  # Zero-width space (remove entirely)

        # Step 3: Clean up multiple spaces that might have been created
        cleaned = re.sub(r' +', ' ', cleaned)

        # Step 4: Strip leading and trailing whitespace
        cleaned = cleaned.strip()

        print(f"Original: {repr(title_text)}")
        print(f"Cleaned:  {repr(cleaned)}")
        print(f"Result:   '{cleaned}'")
        return cleaned if cleaned else 'No title'

    def clean_expiry_date(self, expiry_text: str) -> str:
        """Clean expiry date by removing 'Expiry' prefix and converting to ISO format"""
        if not expiry_text or expiry_text == 'No expiry date found':
            return None

        # Remove 'Expiry' prefix (case insensitive)
        cleaned = re.sub(r'^expiry\s*', '', expiry_text.strip(), flags=re.IGNORECASE)

        # Remove any extra whitespace
        cleaned = cleaned.strip()

        # If nothing left after cleaning, return None
        if not cleaned:
            return None

        # Try to parse and convert common date formats to ISO format (YYYY-MM-DD)
        try:
            # Common date formats to try
            date_formats = [
                '%d/%m/%Y',  # 31/12/2025
                '%d-%m-%Y',  # 31-12-2025
                '%d.%m.%Y',  # 31.12.2025
                '%Y-%m-%d',  # 2025-12-31 (already ISO)
                '%d %B %Y',  # 31 December 2025
                '%d %b %Y',  # 31 Dec 2025
                '%B %d, %Y',  # December 31, 2025
                '%b %d, %Y',  # Dec 31, 2025
            ]

            for date_format in date_formats:
                try:
                    parsed_date = datetime.strptime(cleaned, date_format)
                    # Convert to ISO format (YYYY-MM-DD)
                    iso_date = parsed_date.strftime('%Y-%m-%d')
                    print(f"Successfully converted date '{cleaned}' to ISO format: {iso_date}")
                    return iso_date
                except ValueError:
                    continue

            # If no format matches, log and return None
            print(f"Could not parse date format: '{cleaned}'. Skipping this date.")
            return None

        except Exception as e:
            print(f"Error processing date '{cleaned}': {e}")
            return None

    def categorize_shops_with_gemini(self, shop_names: List[str]) -> Dict[str, str]:
        """Categorize shop names using Google Gemini"""
        try:
            # Create a minimal prompt for cost efficiency
            shop_list = ", ".join(shop_names)

            prompt = f"""Categorize these shop names into one of these categories: Food & Drink, Fashion, Tech, Beauty, Home & Living, Travel, E-commerce.

    Shop names: {shop_list}

    Return only a JSON object mapping each shop name to its category. Example format:
    {{"ShopName1": "Fashion", "ShopName2": "Travel"}}"""

            response = self.model.generate_content(prompt)
            print(response.text)  # Log the raw response for debugging

            # Parse the JSON response
            try:
                # Clean the response text to extract JSON
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                elif response_text.startswith('```'):
                    response_text = response_text.replace('```', '').strip()

                categories = json.loads(response_text)

                print(f"Successfully categorized {len(categories)} shops with Gemini")
                return categories

            except json.JSONDecodeError as e:
                print(f"Error parsing Gemini response as JSON: {e}")
                print(f"Raw response: {response.text}")
                return {}

        except Exception as e:
            print(f"Error categorizing shops with Gemini: {e}")
            return {}

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text using Gemini"""
        try:
            # Clean and prepare text for embedding
            clean_text = ' '.join(text.split())  # Remove extra whitespace
            if not clean_text.strip():
                return None

            # Use Gemini's embedding model directly with genai.embed_content
            result = genai.embed_content(
                model="models/text-embedding-004",  # Use the correct embedding model
                content=clean_text,
                task_type="semantic_similarity"  # Optional: specify the task type
            )

            if result and 'embedding' in result:
                return result['embedding']
            else:
                print(f"Warning: No embedding generated for text: {clean_text[:50]}...")
                return None

        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    async def save_to_supabase_stable(self, data: Dict):
        """Save scraped data to Supabase using stable coupon matching with embeddings"""
        try:
            print('🔗 Connected to Supabase')

            # Step 1: Mark all existing coupons as inactive
            print('🔄 Marking existing coupons as inactive...')
            inactive_result = self.supabase.rpc('mark_all_coupons_inactive').execute()
            inactive_count = inactive_result.data if inactive_result.data else 0
            print(f'📝 Marked {inactive_count} coupons as inactive')

            # Step 2: Get shop categories from Gemini
            shop_names = list(data.keys())
            print(f"🤖 Categorizing {len(shop_names)} shops with Gemini...")
            shop_categories = self.categorize_shops_with_gemini(shop_names)

            # Step 3: Process shops and coupons with upsert strategy
            shops_upserted = 0
            coupons_upserted = 0
            coupons_updated = 0
            coupons_created = 0
            coupons_skipped_duplicates = 0
            coupons_skipped_no_title = 0
            embeddings_generated = 0

            for shop_name, shop_data in data.items():
                category = shop_categories.get(shop_name)
                if not category:
                    print(f"⚠️ Skipping shop '{shop_name}' due to missing category.")
                    continue

                try:
                    # Upsert shop using the new function
                    shop_result = self.supabase.rpc('upsert_shop', {
                        'p_name': shop_name,
                        'p_image_url': shop_data['imageUrl'],
                        'p_category': category
                    }).execute()

                    if shop_result.data:
                        shop_id = shop_result.data
                        shops_upserted += 1
                        print(f"🏪 Upserted shop: {shop_name} -> Category: {category}")

                        # Track processed coupons for this shop to prevent duplicates
                        processed_coupon_keys = set()

                        # Process coupons for this shop
                        for coupon in shop_data['coupons']:
                            # Clean the data
                            cleaned_expiry = self.clean_expiry_date(coupon.get('expiryDate', ''))
                            cleaned_title = self.clean_title_text(coupon.get('title', ''))
                            cleaned_code = coupon.get('code', 'No code').strip()

                            # Skip coupons with "No title found"
                            if cleaned_title == 'No title' or cleaned_title == 'No title found':
                                coupons_skipped_no_title += 1
                                print(f"⚠️ Skipping coupon with no title (Code: {cleaned_code})")
                                continue

                            # Create a unique key for this coupon (shop_id, code, title)
                            coupon_key = (shop_id, cleaned_code.lower(), cleaned_title.lower())

                            # Skip if we've already processed this exact coupon for this shop
                            if coupon_key in processed_coupon_keys:
                                coupons_skipped_duplicates += 1
                                print(f"⚠️ Skipping duplicate coupon: {cleaned_title} (Code: {cleaned_code})")
                                continue

                            # Add to processed set
                            processed_coupon_keys.add(coupon_key)

                            # Generate embedding for the coupon
                            print(f"🧠 Generating embedding for: {cleaned_title}")
                            embedding_text = ' '.join([
                                cleaned_title,
                                coupon.get('description', ''),
                                category,
                                coupon.get('termsAndConditions', '')
                            ]).strip()

                            embedding = await self.generate_embedding(embedding_text)
                            if embedding:
                                embeddings_generated += 1
                                print(f"✅ Generated embedding ({len(embedding)} dimensions)")
                            else:
                                print(f"⚠️ Failed to generate embedding for: {cleaned_title}")

                            # Check if this is an existing coupon by trying to find it first
                            existing_coupon = self.supabase.table('coupons').select('id').eq('shop_id', shop_id).eq(
                                'code', cleaned_code).eq('title', cleaned_title).execute()

                            is_update = existing_coupon.data and len(existing_coupon.data) > 0

                            # Check if coupon is expired
                            is_expired = False
                            if cleaned_expiry:
                                try:
                                    expiry_date_obj = datetime.strptime(cleaned_expiry, '%Y-%m-%d').date()
                                    current_date = datetime.now().date()
                                    is_expired = expiry_date_obj < current_date
                                    if is_expired:
                                        print(f"⏰ Coupon '{cleaned_title}' is expired (expires: {cleaned_expiry})")
                                except Exception as e:
                                    print(f"⚠️ Error parsing expiry date '{cleaned_expiry}': {e}")

                            # Prepare coupon data with embedding
                            coupon_data = {
                                'p_shop_id': shop_id,
                                'p_title': cleaned_title,
                                'p_code': cleaned_code,
                                'p_description': coupon.get('description', 'No description'),
                                'p_terms_and_conditions': coupon.get('termsAndConditions', 'No terms and conditions'),
                                'p_expiry_date': cleaned_expiry,
                                'p_source_url': coupon.get('url', ''),
                                'p_category': category,
                                'p_coupon_image_url': coupon.get('couponImageUrl', ''),
                                'p_embedding': embedding,  # Add embedding to the upsert
                                'p_is_active': not is_expired  # Set active status based on expiry
                            }

                            # Upsert coupon using the enhanced function
                            coupon_result = self.supabase.rpc('upsert_coupon_with_embedding', coupon_data).execute()

                            if coupon_result.data:
                                coupons_upserted += 1
                                if is_update:
                                    coupons_updated += 1
                                    print(f"🔄 Updated existing coupon: {cleaned_title}")
                                else:
                                    coupons_created += 1
                                    print(f"✨ Created new coupon: {cleaned_title}")
                            else:
                                print(f"❌ Failed to upsert coupon: {cleaned_title}")

                            # Add small delay to respect API rate limits
                            await asyncio.sleep(0.1)

                    else:
                        print(f"❌ Failed to upsert shop: {shop_name}")

                except Exception as e:
                    print(f"❌ Error processing shop '{shop_name}': {e}")
                    continue

            # Step 4: Clean up orphaned inactive coupons
            print('🧹 Cleaning up inactive coupons...')
            cleanup_result = self.supabase.rpc('cleanup_inactive_coupons').execute()

            if cleanup_result.data:
                deleted_count = cleanup_result.data[0]['deleted_count']
                preserved_count = cleanup_result.data[0]['preserved_count']
                print(f'🗑️ Deleted {deleted_count} inactive coupons')
                print(f'🔒 Preserved {preserved_count} inactive coupons (user references)')

            # Step 5: Deactivate expired coupons
            print('⏰ Deactivating expired coupons...')
            expired_result = self.supabase.rpc('deactivate_expired_coupons').execute()
            expired_count = 0
            if expired_result.data and len(expired_result.data) > 0:
                expired_count = expired_result.data[0]['deactivated_count']
                print(f'⏰ Deactivated {expired_count} expired coupons')

            # Step 6: Get final statistics
            stats_result = self.supabase.rpc('get_scraping_stats').execute()
            if stats_result.data:
                stats = stats_result.data[0]
                print(f"""
    📊 SCRAPING SUMMARY:
       🏪 Shops processed: {shops_upserted}
       � Couplons processed: {coupons_upserted}
       ✨ New coupons: {coupons_created}
       🔄 Updated coupons: {coupons_updated}
       🧠 Embeddings generated: {embeddings_generated}
       ⚠️ Skipped duplicates: {coupons_skipped_duplicates}
       � Skeipped no title: {coupons_skipped_no_title}
       ⏰ Expired coupons deactivated: {expired_count}

    📈 DATABASE TOTALS:
       🏪 Total shops: {stats['total_shops']}
       🎫 Total coupons: {stats['total_coupons']}
       ✅ Active coupons: {stats['active_coupons']}
       ❌ Inactive coupons: {stats['inactive_coupons']}
       👤 User saved (public): {stats['user_saved_public_coupons']}
       👤 User saved (private): {stats['user_saved_private_coupons']}
                """)

            print('✅ Successfully completed stable coupon matching with embeddings!')

        except Exception as e:
            print(f'❌ Error during stable save to Supabase: {e}')

    async def scrape_coupons(self):
        """Main scraping function"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            context.set_default_navigation_timeout(60000)

            try:
                main_page = await context.new_page()
                print('Navigating to the main page...')
                await main_page.goto('https://www.cuponation.com.my/allshop',
                                     wait_until='networkidle')

                shops_processed = 0

                while shops_processed < self.max_shops:
                    # Get shop links
                    shop_links = await main_page.query_selector_all(
                        'a[class*="188gvwx0"][class*="188gvwx2"][class*="188gvwxs"][class*="188gvwxo"]'
                    )
                    print(f'Found {len(shop_links)} shops on the page')

                    if not shop_links:
                        print('No more shops found, exiting...')
                        break

                    shop_processed_this_iteration = False

                    for shop_link in shop_links:
                        shop_name = await shop_link.inner_text()

                        if shop_name in self.processed_shops:
                            print(f'Skipping already processed shop: {shop_name}')
                            continue

                        print(f'Processing shop {shops_processed + 1}/{self.max_shops}: {shop_name}')
                        self.processed_shops.add(shop_name)
                        shop_processed_this_iteration = True

                        # Get the URL of the shop page
                        shop_url = await shop_link.get_attribute('href')

                        # Extract shop image directly from the shop page
                        print(f'Getting image from shop page: {shop_url}')
                        shop_image_url = ''

                        # Open the shop page first to extract the image
                        preview_page = await context.new_page()
                        try:
                            await preview_page.goto(f'https://www.cuponation.com.my{shop_url}',
                                                    wait_until='networkidle')

                            # Look for images with specific classes
                            image_selectors = [
                                'img[class*="_62by6g0"][class*="_62by6go"][class*="_62by6gp"][class*="_62by6gs"][class*="_62by6gu"]'
                            ]

                            for selector in image_selectors:
                                print(f'Trying image selector on shop page: {selector}')
                                img_element = await preview_page.query_selector(selector)
                                if img_element:
                                    is_visible = await img_element.is_visible()
                                    if is_visible:
                                        shop_image_url = await img_element.get_attribute('src')
                                        print(f'Found shop image on shop page: {shop_image_url}')
                                        break

                            # If still no image, try a more aggressive approach
                            if not shop_image_url:
                                print('Trying to find any relevant image on the shop page...')
                                all_images = await preview_page.query_selector_all('img')
                                print(f'Found {len(all_images)} images on the page')

                                for img in all_images:
                                    is_visible = await img.is_visible()
                                    if not is_visible:
                                        continue

                                    src = await img.get_attribute('src')
                                    alt = await img.get_attribute('alt') or ''

                                    # Look for images that are likely logos
                                    if (src and
                                            (shop_name.lower() in src.lower() or
                                             shop_name.lower() in alt.lower())):
                                        shop_image_url = src
                                        print(f'Found potential logo image: {shop_image_url}')
                                        break

                        except Exception as e:
                            print(f'Error extracting image from shop page: {e}')
                        finally:
                            await preview_page.close()

                        # Initialize array for this shop's coupons
                        self.shop_results[shop_name] = {
                            'imageUrl': shop_image_url,
                            'coupons': []
                        }

                        shop_page = await context.new_page()
                        await shop_page.goto(f'https://www.cuponation.com.my{shop_url}',
                                             wait_until='networkidle')
                        print(f'Navigated to shop page: {shop_page.url}')

                        processed_coupons = set()  # Track coupons for this shop only

                        # Process "See promo code" buttons on "Verified" cards
                        while True:
                            # Fetch all "See promo code" buttons using locator
                            promo_buttons_locator = shop_page.get_by_role('button', name='See promo code')
                            promo_buttons = await promo_buttons_locator.all()
                            print(f'Found {len(promo_buttons)} coupon buttons for {shop_name}')

                            if not promo_buttons:
                                print(f'No more coupons for {shop_name}')
                                break

                            all_processed = True

                            for j in range(len(promo_buttons)):
                                try:
                                    # Re-fetch buttons to avoid stale references
                                    buttons_locator = shop_page.get_by_role('button', name='See promo code')
                                    buttons = await buttons_locator.all()
                                    if len(buttons) <= j:
                                        print(f'Button at index {j} no longer available, skipping...')
                                        continue

                                    button = buttons[j]

                                    # Check if the button's parent card contains "Verified"
                                    # Fixed: Use locator().first() instead of .first
                                    card_locator = button.locator(
                                        'xpath=ancestor::*[contains(@data-testid, "vouchers-ui-voucher-card-top-container")]'
                                    )
                                    card = card_locator.first
                                    card_text = await card.inner_text()
                                    if 'verified' not in card_text.lower():
                                        continue

                                    print(f'Processing verified coupon {j + 1}/{len(promo_buttons)}')

                                    # Extract the code title before clicking
                                    code_title = 'No title found'
                                    try:
                                        title_locator = card.locator(
                                            'div[class*="n9fwq61"][class*="n9fwq65"][class*="n9fwq63"]'
                                        )
                                        title_div = title_locator.first
                                        is_visible = await title_div.is_visible()
                                        if is_visible:
                                            raw_title = await title_div.inner_text()
                                            # Clean the title immediately after extraction
                                            code_title = self.clean_title_text(raw_title)
                                            print(f'Found and cleaned code title: {code_title}')
                                    except Exception as e:
                                        print('Error extracting code title:', e)

                                    # Get the expiry date before clicking the button
                                    expiry_date = 'No expiry date found'
                                    try:
                                        expiry_selectors = [
                                            'div[class*="_7ldhzz0"] span[class*="az57m40"][class*="az57m4c"]',
                                            'span[class*="az57m40"][class*="az57m4c"]'
                                        ]

                                        for selector in expiry_selectors:
                                            print(f'Trying expiry date selector: {selector}')
                                            expiry_locator = card.locator(selector)
                                            expiry_element = expiry_locator.first
                                            try:
                                                is_visible = await expiry_element.is_visible()
                                                if is_visible:
                                                    raw_expiry = await expiry_element.inner_text()
                                                    # Clean the expiry date here during scraping
                                                    expiry_date = self.clean_expiry_date(
                                                        raw_expiry) or 'No expiry date found'
                                                    print(f'Found and cleaned expiry date: {expiry_date}')
                                                    break
                                            except:
                                                continue
                                    except Exception as e:
                                        print('No expiry date found on card:', e)

                                    # Click the button and expect a new page
                                    popup_page = None
                                    try:
                                        async with context.expect_page(timeout=10000) as popup_info:
                                            await button.click()
                                        popup_page = await popup_info.value
                                        await popup_page.wait_for_load_state('networkidle')
                                        print(f'Switched to coupon page: {popup_page.url}')
                                    except Exception as e:
                                        print(f'Error opening popup page: {e}')
                                        continue

                                    # Close the shop page if it's now an ad
                                    if shop_page:
                                        print('Closing shop page (now an ad)...')
                                        await shop_page.close()
                                        shop_page = None

                                    # Extract the promo code immediately
                                    code = 'No code found'
                                    description = 'No description found'
                                    terms_and_conditions = 'No terms and conditions found'
                                    current_url = popup_page.url

                                    # Skip if this coupon was already processed for this shop
                                    if current_url in processed_coupons:
                                        print(f'Coupon at {current_url} already processed, skipping...')
                                        await popup_page.close()
                                        shop_page = await context.new_page()
                                        await shop_page.goto(f'https://www.cuponation.com.my{shop_url}',
                                                             wait_until='networkidle')
                                        continue

                                    try:
                                        # Extract the code
                                        code_element = await popup_page.query_selector(
                                            'h4[class*="az57m40"][class*="az57m46"][class*="b8qpi79"]'
                                        )
                                        if code_element:
                                            code = await code_element.inner_text()
                                            print(f'Found code: {code}')

                                        # Extract the description
                                        description_selectors = [
                                            'h4[class*="az57m40"][class*="az57m46"]',
                                            'div[class*="az57"] h4',
                                            'div[role="dialog"] h4'
                                        ]

                                        for selector in description_selectors:
                                            desc_element = await popup_page.query_selector(selector)
                                            if desc_element:
                                                description = await desc_element.inner_text()
                                                if description and description != code:
                                                    print(f'Found description: {description}')
                                                    break

                                        # Extract Terms and Conditions
                                        print('Looking for Terms and conditions button...')
                                        terms_button = None

                                        # Try to find Terms and Conditions button using text content
                                        try:
                                            # Use get_by_text for more reliable text matching
                                            terms_button = await popup_page.get_by_text('Terms and conditions').first
                                            if terms_button:
                                                print('Found Terms and conditions button!')
                                        except:
                                            # Fallback to query selectors
                                            button_selectors = [
                                                'button:has-text("Terms and conditions")',
                                                'button[class*="ekdz"]'
                                            ]

                                            for selector in button_selectors:
                                                try:
                                                    terms_button = await popup_page.query_selector(selector)
                                                    if terms_button:
                                                        # Verify it contains the text we want
                                                        button_text = await terms_button.inner_text()
                                                        if 'terms and conditions' in button_text.lower():
                                                            print('Found Terms and conditions button!')
                                                            break
                                                        else:
                                                            terms_button = None
                                                except:
                                                    continue

                                        if terms_button:
                                            print('Clicking Terms and conditions button...')
                                            await terms_button.click()
                                            await popup_page.wait_for_timeout(1500)

                                            # Extract terms and conditions content
                                            terms_selectors = [
                                                'div[class*="_1mq6bor0"][class*="_1mq6bor9"][class*="_1mq6bor2"]',
                                                'div[role="dialog"] div p',
                                                'div[aria-modal="true"] div p',
                                                '[role="dialog"] p'
                                            ]

                                            terms_found = False
                                            for selector in terms_selectors:
                                                print(f'Trying terms selector: {selector}')
                                                elements = await popup_page.query_selector_all(selector)
                                                if elements:
                                                    terms_array = []
                                                    for element in elements:
                                                        text = await element.inner_text()
                                                        if text and text.strip():
                                                            terms_array.append(text)

                                                    if terms_array:
                                                        terms_and_conditions = '\n'.join(terms_array)
                                                        print(
                                                            f'Found terms and conditions ({len(terms_array)} paragraphs): {terms_and_conditions[:50]}...')
                                                        terms_found = True
                                                        break

                                            if not terms_found:
                                                print('Unable to find terms and conditions content')

                                            # Try to close the terms modal
                                            close_terms_selectors = [
                                                'button[aria-label="Close"]',
                                                'span[data-testid="CloseIcon"]',
                                                'button:has(svg)',
                                                'button.close-button'
                                            ]

                                            modal_closed = False
                                            for selector in close_terms_selectors:
                                                try:
                                                    close_button = await popup_page.query_selector(selector)
                                                    if close_button:
                                                        await close_button.click()
                                                        await popup_page.wait_for_timeout(500)
                                                        modal_closed = True
                                                        print('Closed terms modal')
                                                        break
                                                except:
                                                    continue

                                            if not modal_closed:
                                                print('Trying to close modal with Escape key')
                                                await popup_page.keyboard.press('Escape')
                                                await popup_page.wait_for_timeout(500)
                                        else:
                                            print('Terms and conditions button not found')

                                    except Exception as e:
                                        print('Error extracting coupon details:', e)

                                    # Add the coupon to the shop's results
                                    self.shop_results[shop_name]['coupons'].append({
                                        'title': code_title,
                                        'code': code,
                                        'description': description,
                                        'termsAndConditions': terms_and_conditions,
                                        'expiryDate': expiry_date,
                                        'url': current_url
                                    })

                                    processed_coupons.add(current_url)
                                    all_processed = False

                                    # Close the popup
                                    try:
                                        close_button = await popup_page.query_selector('span[data-testid="CloseIcon"]')
                                        if close_button:
                                            await close_button.click()
                                        else:
                                            await popup_page.keyboard.press('Escape')
                                        await popup_page.wait_for_timeout(1000)
                                        await popup_page.close()
                                    except:
                                        pass

                                    # Reopen the shop page for the next coupon
                                    shop_page = await context.new_page()
                                    await shop_page.goto(f'https://www.cuponation.com.my{shop_url}',
                                                         wait_until='networkidle')
                                    print(f'Returned to shop page: {shop_page.url}')

                                except Exception as e:
                                    print(f'Error processing coupon {j + 1} for {shop_name}: {e}')
                                    if shop_page:
                                        await shop_page.close()
                                    shop_page = await context.new_page()
                                    await shop_page.goto(f'https://www.cuponation.com.my{shop_url}',
                                                         wait_until='networkidle')

                            # Exit the loop if all coupons are processed
                            if all_processed:
                                print(f'All unique verified coupons processed for {shop_name}')
                                break

                        # Close the shop page and move to the next shop
                        if shop_page:
                            await shop_page.close()
                        shops_processed += 1
                        await main_page.bring_to_front()
                        await main_page.goto('https://www.cuponation.com.my/allshop',
                                             wait_until='networkidle')
                        break

                    if not shop_processed_this_iteration:
                        print('No new shops to process, exiting...')
                        break

                # Save to JSON file (with categories)
                with open('cuponation_coupons.json', 'w', encoding='utf-8') as f:
                    json.dump(self.shop_results, f, indent=2, ensure_ascii=False)

                print(f'Scraping completed. Found coupons for {len(self.shop_results)} shops.')

                # Log the number of coupons per shop
                for shop, shop_data in self.shop_results.items():
                    print(f'{shop}: {len(shop_data["coupons"])} coupons')

                # Calculate total coupons
                total_coupons = sum(len(shop_data['coupons']) for shop_data in self.shop_results.values())
                print(f'Total coupons collected: {total_coupons}')

            except Exception as e:
                print(f'An error occurred during scraping: {e}')
            finally:
                await browser.close()

        return self.shop_results


async def main():
    """Main function to run the scraper"""
    scraper = CouponScraper()

    try:
        results = await scraper.scrape_coupons()
        print(f'Scraping finished with data for {len(results)} shops.')

        # Use the new stable save method (Approach 1)
        await scraper.save_to_supabase_stable(results)

    except Exception as e:
        print(f'Scraping failed: {e}')


if __name__ == '__main__':
    asyncio.run(main())