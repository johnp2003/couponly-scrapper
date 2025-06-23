import asyncio
import json
import os
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
        self.max_shops = 500  # Testing with 10 shops only

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

    async def save_to_supabase(self, data: Dict):
        """Save scraped data to Supabase"""
        try:
            print('Connected to Supabase')

            # Clear existing data
            print('Clearing existing data...')

            # Delete all coupons first (due to foreign key constraints)
            delete_coupons_result = self.supabase.table('coupons').delete().neq('id',
                                                                                '00000000-0000-0000-0000-000000000000').execute()
            print(f'Deleted existing coupons: {len(delete_coupons_result.data) if delete_coupons_result.data else 0}')

            # Delete all shops
            delete_shops_result = self.supabase.table('shops').delete().neq('id',
                                                                            '00000000-0000-0000-0000-000000000000').execute()
            print(f'Deleted existing shops: {len(delete_shops_result.data) if delete_shops_result.data else 0}')

            # Get shop names for categorization
            shop_names = list(data.keys())
            print(f"Categorizing {len(shop_names)} shops with Gemini...")

            # Categorize shops using Gemini
            shop_categories = self.categorize_shops_with_gemini(shop_names)

            # Insert shops and coupons
            shops_inserted = 0
            coupons_inserted = 0

            for shop_name, shop_data in data.items():
                # Check if the shop has a category from Gemini response
                category = shop_categories.get(shop_name)

                if category is None:  # If no category found, skip this shop
                    print(f"Skipping shop '{shop_name}' due to missing category.")
                    continue

                try:
                    # Insert shop into shops table
                    shop_insert_data = {
                        'name': shop_name,
                        'image_url': shop_data['imageUrl'],
                        'category': category
                    }

                    shop_result = self.supabase.table('shops').insert(shop_insert_data).execute()

                    if shop_result.data and len(shop_result.data) > 0:
                        shop_id = shop_result.data[0]['id']
                        shops_inserted += 1
                        print(f"Inserted shop: {shop_name} -> Category: {category} (ID: {shop_id})")

                        # Insert coupons for this shop
                        coupon_data_list = []
                        for coupon in shop_data['coupons']:
                            coupon_insert_data = {
                                'shop_id': shop_id,
                                'title': coupon.get('title', 'No title'),
                                'code': coupon.get('code', 'No code'),
                                'description': coupon.get('description', 'No description'),
                                'terms_and_conditions': coupon.get('termsAndConditions', 'No terms and conditions'),
                                'expiry_date': coupon.get('expiryDate', 'No expiry date'),
                                'source_url': coupon.get('url', ''),
                                'category': category,
                                'is_active': True
                            }
                            coupon_data_list.append(coupon_insert_data)

                        # Batch insert coupons if any exist
                        if coupon_data_list:
                            coupons_result = self.supabase.table('coupons').insert(coupon_data_list).execute()

                            if coupons_result.data:
                                coupons_count = len(coupons_result.data)
                                coupons_inserted += coupons_count
                                print(f"Inserted {coupons_count} coupons for {shop_name}")
                            else:
                                print(f"Failed to insert coupons for {shop_name}")
                        else:
                            print(f"No coupons to insert for {shop_name}")
                    else:
                        print(f"Failed to insert shop: {shop_name}")

                except Exception as e:
                    print(f"Error inserting shop '{shop_name}': {e}")
                    continue

            print(f'Successfully saved {shops_inserted} shops and {coupons_inserted} coupons to Supabase')

        except Exception as e:
            print(f'Error saving to Supabase: {e}')

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
                                            code_title = await title_div.inner_text()
                                            print(f'Found code title: {code_title}')
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
                                                    expiry_date = await expiry_element.inner_text()
                                                    print(f'Found expiry date: {expiry_date}')
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

        # Save to Supabase with categorization
        await scraper.save_to_supabase(results)

    except Exception as e:
        print(f'Scraping failed: {e}')


if __name__ == '__main__':
    asyncio.run(main())