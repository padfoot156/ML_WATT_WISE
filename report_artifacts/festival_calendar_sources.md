# Indian Festival and Holiday Calendar Sources Documentation
**WattWise AI — Energy Demand, Supply & Expansion Intelligence**
**Authors:** Pranav Kolhe (A8 / B4 / 54) & Amber Shukla (A8 / B3 / 38)
**Institution:** Shri Ramdeobaba University, Nagpur

## 1. Statutory and Astronomical Sources
The festival calendar implemented in WattWise AI uses verified dates from:
1. **Ministry of Personnel, Public Grievances and Pensions, Government of India**:
   - Official Gazetted and Restricted Holidays Circular for 2024 (F.No.12/2/2023-JCA)
   - Official Gazetted and Restricted Holidays Circular for 2025 (F.No.12/4/2024-JCA)
2. **Positional Astronomy Centre, India Meteorological Department (IMD), Kolkata**:
   - Astronomical Almanac for Indian Solar and Lunar Calendars (Vikram Samvat 2080-2081 and Saka Samvat 1945-1946).
3. **Reserve Bank of India (RBI) Negotiable Instruments Act Section 25 Schedule**:
   - State-specific commercial and bank holiday lists for Maharashtra (Nagpur/Mumbai circles).

## 2. Movable Festival Resolution Methodology
Unlike Gregorian fixed holidays (e.g., Independence Day on August 15, Republic Day on January 26, Gandhi Jayanti on October 2, Christmas on December 25), many Indian festivals follow the Hindu lunisolar calendar (Tithi) or Islamic lunar calendar (Hijri):
- **Diwali (Deepavali / Lakshmi Puja)**:
  - 2024: November 1, 2024 (Kartik Amavasya)
  - Pre-window (Dhanteras): October 29; Post-window (Govardhan Puja, Bhai Dooj): Nov 2-3
- **Holi**:
  - 2024: March 25, 2024 (Phalguna Purnima)
  - 2025: March 14, 2025
- **Ganesh Chaturthi**:
  - 2024: September 7, 2024 (Bhadrapada Shukla Chaturthi) - of paramount significance in Maharashtra
- **Navratri & Dussehra (Vijayadashami)**:
  - 2024: Navratri Start: October 3, 2024; Dussehra: October 12, 2024
- **Eid al-Fitr & Eid al-Adha**:
  - 2024: April 11, 2024 (Shawwal 1) and June 17, 2024 (Dhu al-Hijjah 10)
  - 2025: March 31, 2025 and June 7, 2025

## 3. Empirical Treatment
Features explicitly track event proximity (pre-festival window, post-festival window, and signed day distance). This captures preparatory industrial ramp-downs, commercial lighting surges, and post-holiday rebound effects observed in National Load Despatch Centre (NLDC) telemetry.
