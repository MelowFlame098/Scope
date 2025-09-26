import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  StyleSheet,
  Dimensions,
} from 'react-native';
import { LineChart } from 'react-native-chart-kit';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useAppSelector, useAppDispatch } from '../../store';
import { useTheme } from '../../context/ThemeContext';
import { useWebSocket } from '../../context/WebSocketContext';
import { Card } from '../../components/common/Card';
import { Button } from '../../components/common/Button';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { PriceChangeIndicator } from '../../components/common/PriceChangeIndicator';
import { OrderBookWidget } from '../../components/trading/OrderBookWidget';
import { TradingChart } from '../../components/trading/TradingChart';
import { AssetSearchModal } from '../../components/trading/AssetSearchModal';
import { OrderConfirmationModal } from '../../components/trading/OrderConfirmationModal';
import { placeOrder, fetchActiveOrders } from '../../store/slices/tradingSlice';
import { fetchAssetPrice, subscribeToAsset } from '../../store/slices/marketDataSlice';

const { width: screenWidth } = Dimensions.get('window');

type OrderType = 'market' | 'limit' | 'stop' | 'stop-limit';
type OrderSide = 'buy' | 'sell';

interface OrderForm {
  symbol: string;
  side: OrderSide;
  type: OrderType;
  quantity: string;
  price: string;
  stopPrice: string;
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'DAY';
}

export const TradingScreen: React.FC = () => {
  const dispatch = useAppDispatch();
  const { theme } = useTheme();
  const { isConnected } = useWebSocket();
  
  const [selectedAsset, setSelectedAsset] = useState('AAPL');
  const [orderForm, setOrderForm] = useState<OrderForm>({
    symbol: 'AAPL',
    side: 'buy',
    type: 'market',
    quantity: '',
    price: '',
    stopPrice: '',
    timeInForce: 'GTC',
  });
  const [showAssetSearch, setShowAssetSearch] = useState(false);
  const [showOrderConfirmation, setShowOrderConfirmation] = useState(false);
  const [activeTab, setActiveTab] = useState<'chart' | 'orderbook' | 'orders'>('chart');

  const {
    currentPrice,
    priceData,
    activeOrders,
    isLoading,
    error,
  } = useAppSelector((state) => ({
    currentPrice: state.marketData.prices[selectedAsset],
    priceData: state.marketData.chartData[selectedAsset],
    activeOrders: state.trading.activeOrders,
    isLoading: state.trading.isLoading,
    error: state.trading.error,
  }));

  const loadAssetData = useCallback(async (symbol: string) => {
    try {
      await dispatch(fetchAssetPrice(symbol)).unwrap();
      dispatch(subscribeToAsset(symbol));
    } catch (error) {
      console.error('Failed to load asset data:', error);
    }
  }, [dispatch]);

  useEffect(() => {
    loadAssetData(selectedAsset);
    dispatch(fetchActiveOrders());
  }, [selectedAsset, loadAssetData, dispatch]);

  const handleAssetSelect = (symbol: string, name: string) => {
    setSelectedAsset(symbol);
    setOrderForm(prev => ({ ...prev, symbol }));
    setShowAssetSearch(false);
    loadAssetData(symbol);
  };

  const handleOrderSubmit = async () => {
    try {
      // Validate form
      if (!orderForm.quantity || parseFloat(orderForm.quantity) <= 0) {
        Alert.alert('Error', 'Please enter a valid quantity');
        return;
      }

      if (orderForm.type !== 'market' && (!orderForm.price || parseFloat(orderForm.price) <= 0)) {
        Alert.alert('Error', 'Please enter a valid price');
        return;
      }

      setShowOrderConfirmation(true);
    } catch (error) {
      Alert.alert('Error', 'Failed to validate order');
    }
  };

  const confirmOrder = async () => {
    try {
      await dispatch(placeOrder({
        symbol: orderForm.symbol,
        side: orderForm.side,
        type: orderForm.type,
        quantity: parseFloat(orderForm.quantity),
        price: orderForm.price ? parseFloat(orderForm.price) : undefined,
        stopPrice: orderForm.stopPrice ? parseFloat(orderForm.stopPrice) : undefined,
        timeInForce: orderForm.timeInForce,
      })).unwrap();

      setShowOrderConfirmation(false);
      setOrderForm(prev => ({ ...prev, quantity: '', price: '', stopPrice: '' }));
      Alert.alert('Success', 'Order placed successfully');
    } catch (error: any) {
      Alert.alert('Error', error.message || 'Failed to place order');
    }
  };

  const calculateOrderValue = () => {
    const quantity = parseFloat(orderForm.quantity) || 0;
    const price = orderForm.type === 'market' 
      ? currentPrice?.price || 0 
      : parseFloat(orderForm.price) || 0;
    return quantity * price;
  };

  const orderTypes = [
    { key: 'market', label: 'Market' },
    { key: 'limit', label: 'Limit' },
    { key: 'stop', label: 'Stop' },
    { key: 'stop-limit', label: 'Stop Limit' },
  ];

  const timeInForceOptions = [
    { key: 'GTC', label: 'Good Till Canceled' },
    { key: 'IOC', label: 'Immediate or Cancel' },
    { key: 'FOK', label: 'Fill or Kill' },
    { key: 'DAY', label: 'Day Order' },
  ];

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.assetSelector}
          onPress={() => setShowAssetSearch(true)}
        >
          <Text style={[styles.assetSymbol, { color: theme.colors.text }]}>
            {selectedAsset}
          </Text>
          <Icon name="keyboard-arrow-down" size={20} color={theme.colors.textSecondary} />
        </TouchableOpacity>
        
        <View style={styles.priceInfo}>
          <Text style={[styles.currentPrice, { color: theme.colors.text }]}>
            ${currentPrice?.price?.toFixed(2) || '0.00'}
          </Text>
          {currentPrice && (
            <PriceChangeIndicator
              value={currentPrice.change}
              percentage={currentPrice.changePercent}
              showIcon
            />
          )}
        </View>

        <View style={styles.connectionStatus}>
          <View
            style={[
              styles.connectionDot,
              { backgroundColor: isConnected ? theme.colors.success : theme.colors.error },
            ]}
          />
        </View>
      </View>

      {/* Tab Navigation */}
      <View style={styles.tabContainer}>
        {[{ key: 'chart', label: 'Chart' }, { key: 'orderbook', label: 'Order Book' }, { key: 'orders', label: 'Orders' }].map((tab) => (
          <TouchableOpacity
            key={tab.key}
            style={[
              styles.tab,
              activeTab === tab.key && { borderBottomColor: theme.colors.primary },
            ]}
            onPress={() => setActiveTab(tab.key as any)}
          >
            <Text
              style={[
                styles.tabText,
                {
                  color: activeTab === tab.key ? theme.colors.primary : theme.colors.textSecondary,
                },
              ]}
            >
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Chart/OrderBook/Orders Content */}
        {activeTab === 'chart' && (
          <Card style={styles.chartCard}>
            <TradingChart symbol={selectedAsset} data={priceData} />
          </Card>
        )}

        {activeTab === 'orderbook' && (
          <Card style={styles.orderBookCard}>
            <OrderBookWidget symbol={selectedAsset} />
          </Card>
        )}

        {activeTab === 'orders' && (
          <Card style={styles.ordersCard}>
            <Text style={[styles.ordersTitle, { color: theme.colors.text }]}>Active Orders</Text>
            {activeOrders.length === 0 ? (
              <View style={styles.emptyOrders}>
                <Icon name="receipt" size={48} color={theme.colors.textSecondary} />
                <Text style={[styles.emptyOrdersText, { color: theme.colors.textSecondary }]}>
                  No active orders
                </Text>
              </View>
            ) : (
              activeOrders.map((order) => (
                <View key={order.id} style={styles.orderItem}>
                  <View style={styles.orderHeader}>
                    <Text style={[styles.orderSymbol, { color: theme.colors.text }]}>
                      {order.symbol}
                    </Text>
                    <Text
                      style={[
                        styles.orderSide,
                        { color: order.side === 'buy' ? theme.colors.success : theme.colors.error },
                      ]}
                    >
                      {order.side.toUpperCase()}
                    </Text>
                  </View>
                  <View style={styles.orderDetails}>
                    <Text style={[styles.orderDetail, { color: theme.colors.textSecondary }]}>
                      {order.type} • {order.quantity} @ ${order.price}
                    </Text>
                    <Text style={[styles.orderStatus, { color: theme.colors.warning }]}>
                      {order.status}
                    </Text>
                  </View>
                </View>
              ))
            )}
          </Card>
        )}

        {/* Order Form */}
        <Card style={styles.orderForm}>
          <Text style={[styles.orderFormTitle, { color: theme.colors.text }]}>Place Order</Text>
          
          {/* Buy/Sell Toggle */}
          <View style={styles.sideToggle}>
            <TouchableOpacity
              style={[
                styles.sideButton,
                orderForm.side === 'buy' && { backgroundColor: theme.colors.success },
              ]}
              onPress={() => setOrderForm(prev => ({ ...prev, side: 'buy' }))}
            >
              <Text
                style={[
                  styles.sideButtonText,
                  { color: orderForm.side === 'buy' ? 'white' : theme.colors.success },
                ]}
              >
                BUY
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[
                styles.sideButton,
                orderForm.side === 'sell' && { backgroundColor: theme.colors.error },
              ]}
              onPress={() => setOrderForm(prev => ({ ...prev, side: 'sell' }))}
            >
              <Text
                style={[
                  styles.sideButtonText,
                  { color: orderForm.side === 'sell' ? 'white' : theme.colors.error },
                ]}
              >
                SELL
              </Text>
            </TouchableOpacity>
          </View>

          {/* Order Type */}
          <View style={styles.orderTypeContainer}>
            <Text style={[styles.inputLabel, { color: theme.colors.text }]}>Order Type</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              <View style={styles.orderTypeButtons}>
                {orderTypes.map((type) => (
                  <TouchableOpacity
                    key={type.key}
                    style={[
                      styles.orderTypeButton,
                      orderForm.type === type.key && { backgroundColor: theme.colors.primary },
                    ]}
                    onPress={() => setOrderForm(prev => ({ ...prev, type: type.key as OrderType }))}
                  >
                    <Text
                      style={[
                        styles.orderTypeButtonText,
                        {
                          color: orderForm.type === type.key ? 'white' : theme.colors.textSecondary,
                        },
                      ]}
                    >
                      {type.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </ScrollView>
          </View>

          {/* Quantity */}
          <View style={styles.inputContainer}>
            <Text style={[styles.inputLabel, { color: theme.colors.text }]}>Quantity</Text>
            <TextInput
              style={[
                styles.input,
                {
                  backgroundColor: theme.colors.surface,
                  color: theme.colors.text,
                  borderColor: theme.colors.border,
                },
              ]}
              value={orderForm.quantity}
              onChangeText={(text) => setOrderForm(prev => ({ ...prev, quantity: text }))}
              placeholder="0"
              placeholderTextColor={theme.colors.textSecondary}
              keyboardType="numeric"
            />
          </View>

          {/* Price (for limit orders) */}
          {orderForm.type !== 'market' && (
            <View style={styles.inputContainer}>
              <Text style={[styles.inputLabel, { color: theme.colors.text }]}>Price</Text>
              <TextInput
                style={[
                  styles.input,
                  {
                    backgroundColor: theme.colors.surface,
                    color: theme.colors.text,
                    borderColor: theme.colors.border,
                  },
                ]}
                value={orderForm.price}
                onChangeText={(text) => setOrderForm(prev => ({ ...prev, price: text }))}
                placeholder="0.00"
                placeholderTextColor={theme.colors.textSecondary}
                keyboardType="numeric"
              />
            </View>
          )}

          {/* Stop Price (for stop orders) */}
          {(orderForm.type === 'stop' || orderForm.type === 'stop-limit') && (
            <View style={styles.inputContainer}>
              <Text style={[styles.inputLabel, { color: theme.colors.text }]}>Stop Price</Text>
              <TextInput
                style={[
                  styles.input,
                  {
                    backgroundColor: theme.colors.surface,
                    color: theme.colors.text,
                    borderColor: theme.colors.border,
                  },
                ]}
                value={orderForm.stopPrice}
                onChangeText={(text) => setOrderForm(prev => ({ ...prev, stopPrice: text }))}
                placeholder="0.00"
                placeholderTextColor={theme.colors.textSecondary}
                keyboardType="numeric"
              />
            </View>
          )}

          {/* Order Summary */}
          <View style={styles.orderSummary}>
            <View style={styles.summaryRow}>
              <Text style={[styles.summaryLabel, { color: theme.colors.textSecondary }]}>Estimated Value</Text>
              <Text style={[styles.summaryValue, { color: theme.colors.text }]}>
                ${calculateOrderValue().toFixed(2)}
              </Text>
            </View>
            <View style={styles.summaryRow}>
              <Text style={[styles.summaryLabel, { color: theme.colors.textSecondary }]}>Estimated Fee</Text>
              <Text style={[styles.summaryValue, { color: theme.colors.text }]}>
                ${(calculateOrderValue() * 0.001).toFixed(2)}
              </Text>
            </View>
          </View>

          {/* Submit Button */}
          <Button
            title={`${orderForm.side.toUpperCase()} ${selectedAsset}`}
            onPress={handleOrderSubmit}
            loading={isLoading}
            style={[
              styles.submitButton,
              {
                backgroundColor: orderForm.side === 'buy' ? theme.colors.success : theme.colors.error,
              },
            ]}
          />
        </Card>
      </ScrollView>

      {/* Modals */}
      <AssetSearchModal
        visible={showAssetSearch}
        onClose={() => setShowAssetSearch(false)}
        onSelect={handleAssetSelect}
      />

      <OrderConfirmationModal
        visible={showOrderConfirmation}
        onClose={() => setShowOrderConfirmation(false)}
        onConfirm={confirmOrder}
        order={{
          ...orderForm,
          quantity: parseFloat(orderForm.quantity) || 0,
          price: parseFloat(orderForm.price) || currentPrice?.price || 0,
          estimatedValue: calculateOrderValue(),
        }}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  assetSelector: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  assetSymbol: {
    fontSize: 18,
    fontWeight: 'bold',
    marginRight: 4,
  },
  priceInfo: {
    alignItems: 'center',
  },
  currentPrice: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  connectionStatus: {
    alignItems: 'center',
  },
  connectionDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  tabContainer: {
    flexDirection: 'row',
    paddingHorizontal: 20,
  },
  tab: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
    borderBottomWidth: 2,
    borderBottomColor: 'transparent',
  },
  tabText: {
    fontSize: 14,
    fontWeight: '500',
  },
  content: {
    flex: 1,
  },
  chartCard: {
    margin: 20,
    padding: 0,
  },
  orderBookCard: {
    margin: 20,
    padding: 16,
  },
  ordersCard: {
    margin: 20,
    padding: 16,
  },
  ordersTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 16,
  },
  emptyOrders: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  emptyOrdersText: {
    marginTop: 12,
    fontSize: 14,
  },
  orderItem: {
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.1)',
  },
  orderHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  orderSymbol: {
    fontSize: 14,
    fontWeight: '600',
  },
  orderSide: {
    fontSize: 12,
    fontWeight: 'bold',
  },
  orderDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  orderDetail: {
    fontSize: 12,
  },
  orderStatus: {
    fontSize: 12,
    fontWeight: '500',
  },
  orderForm: {
    margin: 20,
    padding: 20,
  },
  orderFormTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 20,
  },
  sideToggle: {
    flexDirection: 'row',
    marginBottom: 20,
  },
  sideButton: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
    borderRadius: 8,
    marginHorizontal: 4,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  sideButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  orderTypeContainer: {
    marginBottom: 20,
  },
  orderTypeButtons: {
    flexDirection: 'row',
  },
  orderTypeButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 16,
    marginRight: 8,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  orderTypeButtonText: {
    fontSize: 12,
    fontWeight: '500',
  },
  inputContainer: {
    marginBottom: 16,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 8,
  },
  input: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 8,
    borderWidth: 1,
    fontSize: 16,
  },
  orderSummary: {
    marginVertical: 20,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.1)',
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  summaryLabel: {
    fontSize: 14,
  },
  summaryValue: {
    fontSize: 14,
    fontWeight: '600',
  },
  submitButton: {
    marginTop: 16,
  },
});