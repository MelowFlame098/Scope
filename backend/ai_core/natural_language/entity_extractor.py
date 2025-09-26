# Entity Extractor
# Phase 9: AI-First Platform Implementation

import logging
import asyncio
import re
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
from dateutil import parser as date_parser
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans

logger = logging.getLogger(__name__)

class EntityType(Enum):
    STOCK_SYMBOL = "STOCK_SYMBOL"
    CURRENCY = "CURRENCY"
    PERCENTAGE = "PERCENTAGE"
    MONEY = "MONEY"
    QUANTITY = "QUANTITY"
    DATE = "DATE"
    TIME = "TIME"
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    FINANCIAL_INSTRUMENT = "FINANCIAL_INSTRUMENT"
    MARKET_INDEX = "MARKET_INDEX"
    SECTOR = "SECTOR"
    EXCHANGE = "EXCHANGE"
    ORDER_TYPE = "ORDER_TYPE"
    ACTION = "ACTION"
    PRICE_LEVEL = "PRICE_LEVEL"
    DURATION = "DURATION"
    RISK_METRIC = "RISK_METRIC"
    TECHNICAL_INDICATOR = "TECHNICAL_INDICATOR"

@dataclass
class ExtractedEntity:
    text: str
    label: EntityType
    start: int
    end: int
    confidence: float
    normalized_value: Optional[Any] = None
    metadata: Dict[str, Any] = None

@dataclass
class EntityExtractionResult:
    entities: List[ExtractedEntity]
    extraction_time: float
    methods_used: List[str]
    confidence_scores: Dict[str, float]

class EntityExtractor:
    """Advanced entity extractor for financial conversations"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self.matcher = None
        
        # Financial entity patterns
        self.stock_symbols = self._load_stock_symbols()
        self.currency_codes = self._load_currency_codes()
        self.market_indices = self._load_market_indices()
        self.sectors = self._load_sectors()
        self.exchanges = self._load_exchanges()
        self.order_types = self._load_order_types()
        self.actions = self._load_actions()
        self.risk_metrics = self._load_risk_metrics()
        self.technical_indicators = self._load_technical_indicators()
        
        # Regex patterns
        self.patterns = self._compile_patterns()
        
        # Initialize NLP model
        asyncio.create_task(self._initialize_nlp())
        
        logger.info("Entity extractor initialized")
    
    async def extract(self, processed_query: Any) -> EntityExtractionResult:
        """Extract entities from processed query"""
        start_time = datetime.now()
        
        try:
            text = processed_query.cleaned_text
            tokens = processed_query.tokens
            
            # Extract entities using multiple methods
            entities = []
            methods_used = []
            confidence_scores = {}
            
            # Method 1: SpaCy NER
            if self.nlp:
                spacy_entities = await self._extract_with_spacy(text)
                entities.extend(spacy_entities)
                methods_used.append("spacy_ner")
                confidence_scores["spacy_ner"] = len(spacy_entities) / max(len(tokens), 1)
            
            # Method 2: Pattern matching
            pattern_entities = await self._extract_with_patterns(text)
            entities.extend(pattern_entities)
            methods_used.append("pattern_matching")
            confidence_scores["pattern_matching"] = len(pattern_entities) / max(len(tokens), 1)
            
            # Method 3: Financial dictionary lookup
            dict_entities = await self._extract_with_dictionaries(text, tokens)
            entities.extend(dict_entities)
            methods_used.append("dictionary_lookup")
            confidence_scores["dictionary_lookup"] = len(dict_entities) / max(len(tokens), 1)
            
            # Method 4: Rule-based extraction
            rule_entities = await self._extract_with_rules(text, tokens)
            entities.extend(rule_entities)
            methods_used.append("rule_based")
            confidence_scores["rule_based"] = len(rule_entities) / max(len(tokens), 1)
            
            # Deduplicate and resolve conflicts
            entities = self._deduplicate_entities(entities)
            
            # Normalize entity values
            entities = await self._normalize_entities(entities)
            
            # Validate entities
            entities = self._validate_entities(entities)
            
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            return EntityExtractionResult(
                entities=entities,
                extraction_time=extraction_time,
                methods_used=methods_used,
                confidence_scores=confidence_scores
            )
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            return EntityExtractionResult(
                entities=[],
                extraction_time=extraction_time,
                methods_used=["error"],
                confidence_scores={}
            )
    
    async def _initialize_nlp(self):
        """Initialize SpaCy NLP model"""
        try:
            self.nlp = spacy.load(self.model_name)
            self.matcher = Matcher(self.nlp.vocab)
            
            # Add custom patterns to matcher
            self._add_custom_patterns()
            
            logger.info(f"SpaCy model '{self.model_name}' loaded successfully")
            
        except OSError:
            logger.warning(f"SpaCy model '{self.model_name}' not found. Using fallback methods.")
            self.nlp = None
            self.matcher = None
        except Exception as e:
            logger.error(f"Error initializing SpaCy: {e}")
            self.nlp = None
            self.matcher = None
    
    async def _extract_with_spacy(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using SpaCy NER"""
        entities = []
        
        try:
            if not self.nlp:
                return entities
            
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                entity_type = self._map_spacy_label(ent.label_)
                if entity_type:
                    entities.append(ExtractedEntity(
                        text=ent.text,
                        label=entity_type,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.8,  # SpaCy doesn't provide confidence scores
                        metadata={"spacy_label": ent.label_}
                    ))
            
            # Extract using custom matcher patterns
            if self.matcher:
                matches = self.matcher(doc)
                for match_id, start, end in matches:
                    span = doc[start:end]
                    pattern_name = self.nlp.vocab.strings[match_id]
                    entity_type = self._get_entity_type_from_pattern(pattern_name)
                    
                    if entity_type:
                        entities.append(ExtractedEntity(
                            text=span.text,
                            label=entity_type,
                            start=span.start_char,
                            end=span.end_char,
                            confidence=0.9,
                            metadata={"pattern": pattern_name}
                        ))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in SpaCy extraction: {e}")
            return entities
    
    async def _extract_with_patterns(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using regex patterns"""
        entities = []
        
        try:
            for pattern_name, pattern_info in self.patterns.items():
                regex = pattern_info['regex']
                entity_type = pattern_info['type']
                confidence = pattern_info.get('confidence', 0.7)
                
                for match in re.finditer(regex, text, re.IGNORECASE):
                    entities.append(ExtractedEntity(
                        text=match.group(),
                        label=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        metadata={"pattern": pattern_name}
                    ))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in pattern extraction: {e}")
            return entities
    
    async def _extract_with_dictionaries(self, text: str, tokens: List[str]) -> List[ExtractedEntity]:
        """Extract entities using financial dictionaries"""
        entities = []
        
        try:
            text_lower = text.lower()
            
            # Stock symbols
            for symbol in self.stock_symbols:
                pattern = r'\b' + re.escape(symbol) + r'\b'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(ExtractedEntity(
                        text=match.group(),
                        label=EntityType.STOCK_SYMBOL,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                        metadata={"symbol": symbol.upper()}
                    ))
            
            # Currency codes
            for currency in self.currency_codes:
                pattern = r'\b' + re.escape(currency) + r'\b'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(ExtractedEntity(
                        text=match.group(),
                        label=EntityType.CURRENCY,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,
                        metadata={"currency_code": currency.upper()}
                    ))
            
            # Market indices
            for index in self.market_indices:
                if index.lower() in text_lower:
                    start_pos = text_lower.find(index.lower())
                    entities.append(ExtractedEntity(
                        text=index,
                        label=EntityType.MARKET_INDEX,
                        start=start_pos,
                        end=start_pos + len(index),
                        confidence=0.8,
                        metadata={"index": index}
                    ))
            
            # Sectors
            for sector in self.sectors:
                if sector.lower() in text_lower:
                    start_pos = text_lower.find(sector.lower())
                    entities.append(ExtractedEntity(
                        text=sector,
                        label=EntityType.SECTOR,
                        start=start_pos,
                        end=start_pos + len(sector),
                        confidence=0.7,
                        metadata={"sector": sector}
                    ))
            
            # Exchanges
            for exchange in self.exchanges:
                if exchange.lower() in text_lower:
                    start_pos = text_lower.find(exchange.lower())
                    entities.append(ExtractedEntity(
                        text=exchange,
                        label=EntityType.EXCHANGE,
                        start=start_pos,
                        end=start_pos + len(exchange),
                        confidence=0.8,
                        metadata={"exchange": exchange}
                    ))
            
            # Order types
            for order_type in self.order_types:
                if order_type.lower() in text_lower:
                    start_pos = text_lower.find(order_type.lower())
                    entities.append(ExtractedEntity(
                        text=order_type,
                        label=EntityType.ORDER_TYPE,
                        start=start_pos,
                        end=start_pos + len(order_type),
                        confidence=0.8,
                        metadata={"order_type": order_type}
                    ))
            
            # Actions
            for action in self.actions:
                if action.lower() in text_lower:
                    start_pos = text_lower.find(action.lower())
                    entities.append(ExtractedEntity(
                        text=action,
                        label=EntityType.ACTION,
                        start=start_pos,
                        end=start_pos + len(action),
                        confidence=0.8,
                        metadata={"action": action}
                    ))
            
            # Risk metrics
            for metric in self.risk_metrics:
                if metric.lower() in text_lower:
                    start_pos = text_lower.find(metric.lower())
                    entities.append(ExtractedEntity(
                        text=metric,
                        label=EntityType.RISK_METRIC,
                        start=start_pos,
                        end=start_pos + len(metric),
                        confidence=0.7,
                        metadata={"risk_metric": metric}
                    ))
            
            # Technical indicators
            for indicator in self.technical_indicators:
                if indicator.lower() in text_lower:
                    start_pos = text_lower.find(indicator.lower())
                    entities.append(ExtractedEntity(
                        text=indicator,
                        label=EntityType.TECHNICAL_INDICATOR,
                        start=start_pos,
                        end=start_pos + len(indicator),
                        confidence=0.7,
                        metadata={"technical_indicator": indicator}
                    ))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in dictionary extraction: {e}")
            return entities
    
    async def _extract_with_rules(self, text: str, tokens: List[str]) -> List[ExtractedEntity]:
        """Extract entities using rule-based methods"""
        entities = []
        
        try:
            # Rule 1: Numbers followed by "shares" or "stocks"
            quantity_pattern = r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:shares?|stocks?|units?)'
            for match in re.finditer(quantity_pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    label=EntityType.QUANTITY,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8,
                    normalized_value=float(match.group(1).replace(',', ''))
                ))
            
            # Rule 2: Price levels (support/resistance)
            price_level_pattern = r'(?:support|resistance|target|stop\s*loss)\s*(?:at|of|level)?\s*\$?(\d+(?:\.\d{2})?)'
            for match in re.finditer(price_level_pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    label=EntityType.PRICE_LEVEL,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                    normalized_value=float(match.group(1))
                ))
            
            # Rule 3: Time durations
            duration_pattern = r'(\d+)\s*(days?|weeks?|months?|years?|hours?|minutes?)'
            for match in re.finditer(duration_pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    label=EntityType.DURATION,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                    metadata={"value": int(match.group(1)), "unit": match.group(2)}
                ))
            
            # Rule 4: Financial instruments
            instrument_pattern = r'\b(?:options?|futures?|bonds?|etfs?|mutual\s*funds?|derivatives?)\b'
            for match in re.finditer(instrument_pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    label=EntityType.FINANCIAL_INSTRUMENT,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in rule-based extraction: {e}")
            return entities
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate and overlapping entities"""
        try:
            if not entities:
                return entities
            
            # Sort by start position
            entities.sort(key=lambda x: (x.start, x.end))
            
            # Remove exact duplicates
            unique_entities = []
            seen = set()
            
            for entity in entities:
                key = (entity.text, entity.label, entity.start, entity.end)
                if key not in seen:
                    unique_entities.append(entity)
                    seen.add(key)
            
            # Resolve overlaps (keep higher confidence entity)
            final_entities = []
            i = 0
            
            while i < len(unique_entities):
                current = unique_entities[i]
                overlapping = [current]
                
                # Find all overlapping entities
                j = i + 1
                while j < len(unique_entities):
                    next_entity = unique_entities[j]
                    if self._entities_overlap(current, next_entity):
                        overlapping.append(next_entity)
                        j += 1
                    else:
                        break
                
                # Keep the entity with highest confidence
                best_entity = max(overlapping, key=lambda x: x.confidence)
                final_entities.append(best_entity)
                
                i = j if j > i + 1 else i + 1
            
            return final_entities
            
        except Exception as e:
            logger.error(f"Error deduplicating entities: {e}")
            return entities
    
    def _entities_overlap(self, entity1: ExtractedEntity, entity2: ExtractedEntity) -> bool:
        """Check if two entities overlap"""
        return not (entity1.end <= entity2.start or entity2.end <= entity1.start)
    
    async def _normalize_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Normalize entity values"""
        try:
            for entity in entities:
                if entity.label == EntityType.MONEY:
                    entity.normalized_value = self._normalize_money(entity.text)
                elif entity.label == EntityType.PERCENTAGE:
                    entity.normalized_value = self._normalize_percentage(entity.text)
                elif entity.label == EntityType.DATE:
                    entity.normalized_value = self._normalize_date(entity.text)
                elif entity.label == EntityType.QUANTITY:
                    entity.normalized_value = self._normalize_quantity(entity.text)
                elif entity.label == EntityType.STOCK_SYMBOL:
                    entity.normalized_value = entity.text.upper()
                elif entity.label == EntityType.CURRENCY:
                    entity.normalized_value = entity.text.upper()
            
            return entities
            
        except Exception as e:
            logger.error(f"Error normalizing entities: {e}")
            return entities
    
    def _normalize_money(self, text: str) -> Optional[float]:
        """Normalize money amounts"""
        try:
            # Remove currency symbols and commas
            clean_text = re.sub(r'[^\d.]', '', text)
            return float(clean_text) if clean_text else None
        except:
            return None
    
    def _normalize_percentage(self, text: str) -> Optional[float]:
        """Normalize percentage values"""
        try:
            # Remove % symbol
            clean_text = text.replace('%', '').strip()
            return float(clean_text) / 100 if clean_text else None
        except:
            return None
    
    def _normalize_date(self, text: str) -> Optional[datetime]:
        """Normalize date values"""
        try:
            return date_parser.parse(text)
        except:
            return None
    
    def _normalize_quantity(self, text: str) -> Optional[float]:
        """Normalize quantity values"""
        try:
            # Extract number from text
            numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text)
            if numbers:
                return float(numbers[0].replace(',', ''))
            return None
        except:
            return None
    
    def _validate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Validate extracted entities"""
        validated = []
        
        try:
            for entity in entities:
                # Basic validation
                if not entity.text or not entity.text.strip():
                    continue
                
                # Type-specific validation
                if entity.label == EntityType.STOCK_SYMBOL:
                    if self._is_valid_stock_symbol(entity.text):
                        validated.append(entity)
                elif entity.label == EntityType.CURRENCY:
                    if self._is_valid_currency(entity.text):
                        validated.append(entity)
                elif entity.label == EntityType.PERCENTAGE:
                    if self._is_valid_percentage(entity.text):
                        validated.append(entity)
                elif entity.label == EntityType.MONEY:
                    if self._is_valid_money(entity.text):
                        validated.append(entity)
                else:
                    # For other types, basic validation is sufficient
                    validated.append(entity)
            
            return validated
            
        except Exception as e:
            logger.error(f"Error validating entities: {e}")
            return entities
    
    def _is_valid_stock_symbol(self, text: str) -> bool:
        """Validate stock symbol"""
        # Basic validation: 1-5 uppercase letters
        return bool(re.match(r'^[A-Z]{1,5}$', text.upper()))
    
    def _is_valid_currency(self, text: str) -> bool:
        """Validate currency code"""
        # Basic validation: 3 uppercase letters
        return bool(re.match(r'^[A-Z]{3}$', text.upper()))
    
    def _is_valid_percentage(self, text: str) -> bool:
        """Validate percentage"""
        try:
            value = float(re.sub(r'[^\d.-]', '', text))
            return 0 <= value <= 100
        except:
            return False
    
    def _is_valid_money(self, text: str) -> bool:
        """Validate money amount"""
        try:
            value = float(re.sub(r'[^\d.-]', '', text))
            return value >= 0
        except:
            return False
    
    def _map_spacy_label(self, spacy_label: str) -> Optional[EntityType]:
        """Map SpaCy entity labels to our entity types"""
        mapping = {
            'MONEY': EntityType.MONEY,
            'PERCENT': EntityType.PERCENTAGE,
            'DATE': EntityType.DATE,
            'TIME': EntityType.TIME,
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'CARDINAL': EntityType.QUANTITY
        }
        return mapping.get(spacy_label)
    
    def _get_entity_type_from_pattern(self, pattern_name: str) -> Optional[EntityType]:
        """Get entity type from pattern name"""
        pattern_mapping = {
            'stock_symbol': EntityType.STOCK_SYMBOL,
            'currency': EntityType.CURRENCY,
            'money': EntityType.MONEY,
            'percentage': EntityType.PERCENTAGE,
            'quantity': EntityType.QUANTITY,
            'order_type': EntityType.ORDER_TYPE,
            'action': EntityType.ACTION
        }
        return pattern_mapping.get(pattern_name)
    
    def _add_custom_patterns(self):
        """Add custom patterns to SpaCy matcher"""
        if not self.matcher:
            return
        
        try:
            # Stock symbol patterns
            stock_patterns = [
                [{"TEXT": {"REGEX": r"^[A-Z]{1,5}$"}}],
                [{"LOWER": "ticker"}, {"TEXT": {"REGEX": r"^[A-Z]{1,5}$"}}]
            ]
            self.matcher.add("stock_symbol", stock_patterns)
            
            # Money patterns
            money_patterns = [
                [{"TEXT": "$"}, {"LIKE_NUM": True}],
                [{"LIKE_NUM": True}, {"LOWER": "dollars"}],
                [{"LIKE_NUM": True}, {"LOWER": "usd"}]
            ]
            self.matcher.add("money", money_patterns)
            
            # Percentage patterns
            percentage_patterns = [
                [{"LIKE_NUM": True}, {"TEXT": "%"}],
                [{"LIKE_NUM": True}, {"LOWER": "percent"}]
            ]
            self.matcher.add("percentage", percentage_patterns)
            
            # Order type patterns
            order_patterns = [
                [{"LOWER": "market"}, {"LOWER": "order"}],
                [{"LOWER": "limit"}, {"LOWER": "order"}],
                [{"LOWER": "stop"}, {"LOWER": "loss"}]
            ]
            self.matcher.add("order_type", order_patterns)
            
        except Exception as e:
            logger.error(f"Error adding custom patterns: {e}")
    
    def _compile_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Compile regex patterns for entity extraction"""
        return {
            'money': {
                'regex': r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|usd|USD)',
                'type': EntityType.MONEY,
                'confidence': 0.8
            },
            'percentage': {
                'regex': r'\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*percent',
                'type': EntityType.PERCENTAGE,
                'confidence': 0.9
            },
            'stock_symbol': {
                'regex': r'\b[A-Z]{1,5}\b(?=\s|$|[^A-Za-z])',
                'type': EntityType.STOCK_SYMBOL,
                'confidence': 0.7
            },
            'date': {
                'regex': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b',
                'type': EntityType.DATE,
                'confidence': 0.8
            },
            'time': {
                'regex': r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
                'type': EntityType.TIME,
                'confidence': 0.8
            }
        }
    
    def _load_stock_symbols(self) -> Set[str]:
        """Load common stock symbols"""
        return {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.A', 'BRK.B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'HD', 'MA', 'BAC', 'ABBV', 'PFE', 'KO', 'AVGO',
            'PEP', 'TMO', 'COST', 'MRK', 'WMT', 'ABT', 'ACN', 'NFLX', 'ADBE', 'DHR', 'VZ',
            'NKE', 'LIN', 'TXN', 'CRM', 'NEE', 'RTX', 'QCOM', 'PM', 'UPS', 'LOW', 'ORCL',
            'HON', 'T', 'INTU', 'IBM', 'AMD', 'AMGN', 'SPGI', 'CAT', 'GS', 'ISRG', 'AXP',
            'BKNG', 'DE', 'TJX', 'MDLZ', 'BLK', 'SYK', 'ADP', 'GILD', 'MMM', 'LRCX', 'CVS',
            'TMUS', 'ZTS', 'MO', 'CI', 'SO', 'DUK', 'BSX', 'REGN', 'EQIX', 'PLD', 'SCHW',
            'AON', 'CL', 'ITW', 'APD', 'CME', 'USB', 'MMC', 'EOG', 'ICE', 'FCX', 'NSC',
            'PNC', 'D', 'WM', 'F', 'EMR', 'GD', 'TGT', 'PSA', 'SHW', 'MCO', 'DG', 'COP'
        }
    
    def _load_currency_codes(self) -> Set[str]:
        """Load currency codes"""
        return {
            'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD', 'SEK', 'NOK',
            'DKK', 'PLN', 'CZK', 'HUF', 'RUB', 'CNY', 'HKD', 'SGD', 'KRW', 'INR',
            'BRL', 'MXN', 'ZAR', 'TRY', 'ILS', 'THB', 'MYR', 'IDR', 'PHP', 'VND'
        }
    
    def _load_market_indices(self) -> Set[str]:
        """Load market indices"""
        return {
            'S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000', 'VIX', 'FTSE 100',
            'DAX', 'CAC 40', 'Nikkei 225', 'Hang Seng', 'Shanghai Composite',
            'BSE Sensex', 'KOSPI', 'ASX 200', 'TSX', 'IBEX 35', 'AEX', 'SMI',
            'FTSE MIB', 'OMX Stockholm 30', 'OBX', 'WIG20', 'BUX', 'RTS'
        }
    
    def _load_sectors(self) -> Set[str]:
        """Load sector names"""
        return {
            'Technology', 'Healthcare', 'Financial Services', 'Consumer Discretionary',
            'Communication Services', 'Industrials', 'Consumer Staples', 'Energy',
            'Utilities', 'Real Estate', 'Materials', 'Biotechnology', 'Pharmaceuticals',
            'Software', 'Semiconductors', 'Banking', 'Insurance', 'Retail', 'Automotive',
            'Aerospace', 'Defense', 'Oil & Gas', 'Mining', 'Agriculture', 'Transportation'
        }
    
    def _load_exchanges(self) -> Set[str]:
        """Load exchange names"""
        return {
            'NYSE', 'NASDAQ', 'LSE', 'TSE', 'HKEX', 'SSE', 'SZSE', 'Euronext',
            'Frankfurt', 'Milan', 'Madrid', 'Amsterdam', 'Stockholm', 'Oslo',
            'Copenhagen', 'Warsaw', 'Prague', 'Budapest', 'Moscow', 'Bombay',
            'NSE', 'ASX', 'TSX', 'JSE', 'BOVESPA', 'BMV'
        }
    
    def _load_order_types(self) -> Set[str]:
        """Load order types"""
        return {
            'Market Order', 'Limit Order', 'Stop Loss', 'Stop Limit', 'Trailing Stop',
            'Fill or Kill', 'Immediate or Cancel', 'Good Till Cancelled', 'Day Order',
            'All or None', 'Iceberg Order', 'Hidden Order', 'Bracket Order'
        }
    
    def _load_actions(self) -> Set[str]:
        """Load trading actions"""
        return {
            'Buy', 'Sell', 'Hold', 'Short', 'Cover', 'Exercise', 'Assign',
            'Execute', 'Cancel', 'Modify', 'Place', 'Submit', 'Close', 'Open'
        }
    
    def _load_risk_metrics(self) -> Set[str]:
        """Load risk metrics"""
        return {
            'VaR', 'Value at Risk', 'CVaR', 'Expected Shortfall', 'Beta', 'Alpha',
            'Sharpe Ratio', 'Sortino Ratio', 'Maximum Drawdown', 'Volatility',
            'Standard Deviation', 'Correlation', 'R-squared', 'Tracking Error',
            'Information Ratio', 'Treynor Ratio', 'Calmar Ratio', 'Omega Ratio'
        }
    
    def _load_technical_indicators(self) -> Set[str]:
        """Load technical indicators"""
        return {
            'RSI', 'MACD', 'Moving Average', 'EMA', 'SMA', 'Bollinger Bands',
            'Stochastic', 'Williams %R', 'CCI', 'ADX', 'ATR', 'OBV', 'Volume',
            'Fibonacci', 'Support', 'Resistance', 'Trend Line', 'Chart Pattern',
            'Head and Shoulders', 'Double Top', 'Double Bottom', 'Triangle',
            'Flag', 'Pennant', 'Cup and Handle', 'Wedge', 'Channel'
        }