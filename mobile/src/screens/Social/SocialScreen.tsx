import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  FlatList,
  Image,
  TextInput,
  Modal,
  Alert,
  StyleSheet,
  Dimensions,
} from 'react-native';
import { useSelector, useDispatch } from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { LineChart, PieChart } from 'react-native-chart-kit';
import { RootState } from '../../store';
import { socialTradingService, TraderProfile, SocialPost, TradingStrategy } from '../../services/SocialTradingService';

const { width: screenWidth } = Dimensions.get('window');

interface SocialScreenProps {
  navigation: any;
}

const SocialScreen: React.FC<SocialScreenProps> = ({ navigation }) => {
  const dispatch = useDispatch();
  const { user } = useSelector((state: RootState) => state.auth);
  const { theme } = useSelector((state: RootState) => state.settings);

  const [activeTab, setActiveTab] = useState<'feed' | 'traders' | 'strategies' | 'leaderboard'>('feed');
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showCreatePost, setShowCreatePost] = useState(false);
  const [postContent, setPostContent] = useState('');
  const [postType, setPostType] = useState<'text' | 'trade' | 'analysis' | 'prediction'>('text');

  // Data states
  const [feed, setFeed] = useState<SocialPost[]>([]);
  const [topTraders, setTopTraders] = useState<TraderProfile[]>([]);
  const [strategies, setStrategies] = useState<TradingStrategy[]>([]);
  const [leaderboard, setLeaderboard] = useState<any[]>([]);
  const [followingTraders, setFollowingTraders] = useState<string[]>([]);

  const isDark = theme === 'dark';
  const colors = {
    background: isDark ? '#1a1a1a' : '#ffffff',
    surface: isDark ? '#2d2d2d' : '#f8f9fa',
    text: isDark ? '#ffffff' : '#000000',
    textSecondary: isDark ? '#b0b0b0' : '#666666',
    primary: '#007AFF',
    success: '#34C759',
    danger: '#FF3B30',
    warning: '#FF9500',
    border: isDark ? '#404040' : '#e0e0e0',
  };

  useEffect(() => {
    loadInitialData();
    setupRealTimeUpdates();

    return () => {
      socialTradingService.disconnect();
    };
  }, []);

  const loadInitialData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadFeed(),
        loadTopTraders(),
        loadStrategies(),
        loadLeaderboard(),
        loadFollowingTraders(),
      ]);
    } catch (error) {
      console.error('Failed to load initial data:', error);
    } finally {
      setLoading(false);
    }
  };

  const setupRealTimeUpdates = () => {
    socialTradingService.subscribeToFeed((post: SocialPost) => {
      setFeed(prev => [post, ...prev]);
    });
  };

  const loadFeed = async () => {
    try {
      const feedData = await socialTradingService.getFeed({ limit: 20 });
      setFeed(feedData);
    } catch (error) {
      console.error('Failed to load feed:', error);
    }
  };

  const loadTopTraders = async () => {
    try {
      const traders = await socialTradingService.searchTraders({
        sortBy: 'performance',
        limit: 10,
      });
      setTopTraders(traders);
    } catch (error) {
      console.error('Failed to load top traders:', error);
    }
  };

  const loadStrategies = async () => {
    try {
      const strategiesData = await socialTradingService.getStrategies({
        sortBy: 'performance',
        limit: 10,
      });
      setStrategies(strategiesData);
    } catch (error) {
      console.error('Failed to load strategies:', error);
    }
  };

  const loadLeaderboard = async () => {
    try {
      const leaderboardData = await socialTradingService.getLeaderboard('1m', 'return');
      setLeaderboard(leaderboardData.traders);
    } catch (error) {
      console.error('Failed to load leaderboard:', error);
    }
  };

  const loadFollowingTraders = async () => {
    try {
      if (user?.id) {
        const following = await socialTradingService.getFollowing(user.id);
        setFollowingTraders(following.map(trader => trader.id));
      }
    } catch (error) {
      console.error('Failed to load following traders:', error);
    }
  };

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadInitialData();
    setRefreshing(false);
  }, []);

  const handleFollowTrader = async (traderId: string) => {
    try {
      if (followingTraders.includes(traderId)) {
        await socialTradingService.unfollowTrader(traderId);
        setFollowingTraders(prev => prev.filter(id => id !== traderId));
      } else {
        await socialTradingService.followTrader(traderId);
        setFollowingTraders(prev => [...prev, traderId]);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to update follow status');
    }
  };

  const handleLikePost = async (postId: string) => {
    try {
      await socialTradingService.likePost(postId);
      setFeed(prev => prev.map(post => 
        post.id === postId 
          ? { ...post, likes: post.likes + 1 }
          : post
      ));
    } catch (error) {
      Alert.alert('Error', 'Failed to like post');
    }
  };

  const handleCreatePost = async () => {
    if (!postContent.trim()) return;

    try {
      const newPost = await socialTradingService.createPost({
        authorId: user?.id || '',
        type: postType,
        content: postContent,
        isPublic: true,
        tags: [],
      });
      
      setFeed(prev => [newPost, ...prev]);
      setPostContent('');
      setShowCreatePost(false);
    } catch (error) {
      Alert.alert('Error', 'Failed to create post');
    }
  };

  const renderTabBar = () => (
    <View style={[styles.tabBar, { backgroundColor: colors.surface, borderBottomColor: colors.border }]}>
      {[
        { key: 'feed', label: 'Feed', icon: 'dynamic-feed' },
        { key: 'traders', label: 'Traders', icon: 'people' },
        { key: 'strategies', label: 'Strategies', icon: 'trending-up' },
        { key: 'leaderboard', label: 'Leaderboard', icon: 'leaderboard' },
      ].map((tab) => (
        <TouchableOpacity
          key={tab.key}
          style={[
            styles.tabItem,
            activeTab === tab.key && { borderBottomColor: colors.primary }
          ]}
          onPress={() => setActiveTab(tab.key as any)}
        >
          <Icon 
            name={tab.icon} 
            size={20} 
            color={activeTab === tab.key ? colors.primary : colors.textSecondary} 
          />
          <Text style={[
            styles.tabLabel,
            { color: activeTab === tab.key ? colors.primary : colors.textSecondary }
          ]}>
            {tab.label}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  const renderFeedItem = ({ item }: { item: SocialPost }) => (
    <View style={[styles.feedItem, { backgroundColor: colors.surface, borderColor: colors.border }]}>
      <View style={styles.feedHeader}>
        <Image 
          source={{ uri: item.author.avatar || 'https://via.placeholder.com/40' }}
          style={styles.avatar}
        />
        <View style={styles.feedAuthor}>
          <Text style={[styles.authorName, { color: colors.text }]}>
            {item.author.displayName}
          </Text>
          <Text style={[styles.feedTime, { color: colors.textSecondary }]}>
            {new Date(item.createdAt).toLocaleDateString()}
          </Text>
        </View>
        {item.author.verified && (
          <Icon name="verified" size={16} color={colors.primary} />
        )}
      </View>
      
      <Text style={[styles.feedContent, { color: colors.text }]}>
        {item.content}
      </Text>
      
      {item.tradeData && (
        <View style={[styles.tradeData, { backgroundColor: colors.background, borderColor: colors.border }]}>
          <Text style={[styles.tradeSymbol, { color: colors.text }]}>
            {item.tradeData.symbol}
          </Text>
          <Text style={[
            styles.tradeSide,
            { color: item.tradeData.side === 'buy' ? colors.success : colors.danger }
          ]}>
            {item.tradeData.side.toUpperCase()}
          </Text>
          <Text style={[styles.tradePrice, { color: colors.text }]}>
            ${item.tradeData.price.toFixed(2)}
          </Text>
        </View>
      )}
      
      <View style={styles.feedActions}>
        <TouchableOpacity 
          style={styles.actionButton}
          onPress={() => handleLikePost(item.id)}
        >
          <Icon name="favorite-border" size={20} color={colors.textSecondary} />
          <Text style={[styles.actionText, { color: colors.textSecondary }]}>
            {item.likes}
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.actionButton}>
          <Icon name="comment" size={20} color={colors.textSecondary} />
          <Text style={[styles.actionText, { color: colors.textSecondary }]}>
            {item.comments}
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.actionButton}>
          <Icon name="share" size={20} color={colors.textSecondary} />
          <Text style={[styles.actionText, { color: colors.textSecondary }]}>
            {item.shares}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderTraderItem = ({ item }: { item: TraderProfile }) => (
    <View style={[styles.traderItem, { backgroundColor: colors.surface, borderColor: colors.border }]}>
      <View style={styles.traderHeader}>
        <Image 
          source={{ uri: item.avatar || 'https://via.placeholder.com/50' }}
          style={styles.traderAvatar}
        />
        <View style={styles.traderInfo}>
          <View style={styles.traderNameRow}>
            <Text style={[styles.traderName, { color: colors.text }]}>
              {item.displayName}
            </Text>
            {item.verified && (
              <Icon name="verified" size={16} color={colors.primary} />
            )}
          </View>
          <Text style={[styles.traderTier, { color: colors.textSecondary }]}>
            {item.tier.toUpperCase()} • {item.followers} followers
          </Text>
        </View>
        <TouchableOpacity
          style={[
            styles.followButton,
            {
              backgroundColor: followingTraders.includes(item.id) ? colors.surface : colors.primary,
              borderColor: colors.primary,
            }
          ]}
          onPress={() => handleFollowTrader(item.id)}
        >
          <Text style={[
            styles.followButtonText,
            { color: followingTraders.includes(item.id) ? colors.primary : '#ffffff' }
          ]}>
            {followingTraders.includes(item.id) ? 'Following' : 'Follow'}
          </Text>
        </TouchableOpacity>
      </View>
      
      <View style={styles.traderStats}>
        <View style={styles.statItem}>
          <Text style={[styles.statValue, { color: colors.success }]}>+24.5%</Text>
          <Text style={[styles.statLabel, { color: colors.textSecondary }]}>Return</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={[styles.statValue, { color: colors.text }]}>1.8</Text>
          <Text style={[styles.statLabel, { color: colors.textSecondary }]}>Sharpe</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={[styles.statValue, { color: colors.text }]}>-8.2%</Text>
          <Text style={[styles.statLabel, { color: colors.textSecondary }]}>Max DD</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={[styles.statValue, { color: colors.text }]}>68%</Text>
          <Text style={[styles.statLabel, { color: colors.textSecondary }]}>Win Rate</Text>
        </View>
      </View>
    </View>
  );

  const renderStrategyItem = ({ item }: { item: TradingStrategy }) => (
    <View style={[styles.strategyItem, { backgroundColor: colors.surface, borderColor: colors.border }]}>
      <View style={styles.strategyHeader}>
        <Text style={[styles.strategyName, { color: colors.text }]}>
          {item.name}
        </Text>
        <View style={[styles.riskBadge, { backgroundColor: getRiskColor(item.riskLevel) }]}>
          <Text style={styles.riskText}>{item.riskLevel.toUpperCase()}</Text>
        </View>
      </View>
      
      <Text style={[styles.strategyDescription, { color: colors.textSecondary }]}>
        {item.description}
      </Text>
      
      <View style={styles.strategyStats}>
        <View style={styles.statRow}>
          <Text style={[styles.statLabel, { color: colors.textSecondary }]}>Return:</Text>
          <Text style={[styles.statValue, { color: colors.success }]}>+18.7%</Text>
        </View>
        <View style={styles.statRow}>
          <Text style={[styles.statLabel, { color: colors.textSecondary }]}>Subscribers:</Text>
          <Text style={[styles.statValue, { color: colors.text }]}>{item.subscribers}</Text>
        </View>
        <View style={styles.statRow}>
          <Text style={[styles.statLabel, { color: colors.textSecondary }]}>AUM:</Text>
          <Text style={[styles.statValue, { color: colors.text }]}>${(item.totalAUM / 1000000).toFixed(1)}M</Text>
        </View>
      </View>
      
      <TouchableOpacity style={[styles.subscribeButton, { backgroundColor: colors.primary }]}>
        <Text style={styles.subscribeButtonText}>Subscribe</Text>
      </TouchableOpacity>
    </View>
  );

  const renderLeaderboardItem = ({ item, index }: { item: any; index: number }) => (
    <View style={[styles.leaderboardItem, { backgroundColor: colors.surface, borderColor: colors.border }]}>
      <View style={styles.rankContainer}>
        <Text style={[styles.rank, { color: colors.text }]}>#{index + 1}</Text>
      </View>
      
      <Image 
        source={{ uri: item.trader.avatar || 'https://via.placeholder.com/40' }}
        style={styles.leaderboardAvatar}
      />
      
      <View style={styles.leaderboardInfo}>
        <Text style={[styles.leaderboardName, { color: colors.text }]}>
          {item.trader.displayName}
        </Text>
        <Text style={[styles.leaderboardTier, { color: colors.textSecondary }]}>
          {item.trader.tier.toUpperCase()}
        </Text>
      </View>
      
      <View style={styles.leaderboardValue}>
        <Text style={[styles.valueText, { color: colors.success }]}>
          +{item.value.toFixed(1)}%
        </Text>
        <Text style={[styles.changeText, { color: item.change >= 0 ? colors.success : colors.danger }]}>
          {item.change >= 0 ? '+' : ''}{item.change.toFixed(1)}%
        </Text>
      </View>
    </View>
  );

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return colors.success;
      case 'medium': return colors.warning;
      case 'high': return colors.danger;
      default: return colors.textSecondary;
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'feed':
        return (
          <FlatList
            data={feed}
            renderItem={renderFeedItem}
            keyExtractor={(item) => item.id}
            refreshControl={
              <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
            }
            showsVerticalScrollIndicator={false}
          />
        );
      
      case 'traders':
        return (
          <FlatList
            data={topTraders}
            renderItem={renderTraderItem}
            keyExtractor={(item) => item.id}
            refreshControl={
              <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
            }
            showsVerticalScrollIndicator={false}
          />
        );
      
      case 'strategies':
        return (
          <FlatList
            data={strategies}
            renderItem={renderStrategyItem}
            keyExtractor={(item) => item.id}
            refreshControl={
              <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
            }
            showsVerticalScrollIndicator={false}
          />
        );
      
      case 'leaderboard':
        return (
          <FlatList
            data={leaderboard}
            renderItem={renderLeaderboardItem}
            keyExtractor={(item, index) => `${item.trader.id}-${index}`}
            refreshControl={
              <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
            }
            showsVerticalScrollIndicator={false}
          />
        );
      
      default:
        return null;
    }
  };

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      {renderTabBar()}
      
      {renderContent()}
      
      {activeTab === 'feed' && (
        <TouchableOpacity
          style={[styles.fab, { backgroundColor: colors.primary }]}
          onPress={() => setShowCreatePost(true)}
        >
          <Icon name="add" size={24} color="#ffffff" />
        </TouchableOpacity>
      )}
      
      <Modal
        visible={showCreatePost}
        animationType="slide"
        presentationStyle="pageSheet"
      >
        <View style={[styles.modalContainer, { backgroundColor: colors.background }]}>
          <View style={[styles.modalHeader, { borderBottomColor: colors.border }]}>
            <TouchableOpacity onPress={() => setShowCreatePost(false)}>
              <Text style={[styles.modalCancel, { color: colors.primary }]}>Cancel</Text>
            </TouchableOpacity>
            <Text style={[styles.modalTitle, { color: colors.text }]}>Create Post</Text>
            <TouchableOpacity onPress={handleCreatePost}>
              <Text style={[styles.modalPost, { color: colors.primary }]}>Post</Text>
            </TouchableOpacity>
          </View>
          
          <View style={styles.postTypeSelector}>
            {[
              { key: 'text', label: 'Text', icon: 'text-fields' },
              { key: 'trade', label: 'Trade', icon: 'trending-up' },
              { key: 'analysis', label: 'Analysis', icon: 'analytics' },
              { key: 'prediction', label: 'Prediction', icon: 'psychology' },
            ].map((type) => (
              <TouchableOpacity
                key={type.key}
                style={[
                  styles.postTypeButton,
                  {
                    backgroundColor: postType === type.key ? colors.primary : colors.surface,
                    borderColor: colors.border,
                  }
                ]}
                onPress={() => setPostType(type.key as any)}
              >
                <Icon 
                  name={type.icon} 
                  size={20} 
                  color={postType === type.key ? '#ffffff' : colors.textSecondary} 
                />
                <Text style={[
                  styles.postTypeLabel,
                  { color: postType === type.key ? '#ffffff' : colors.textSecondary }
                ]}>
                  {type.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
          
          <TextInput
            style={[
              styles.postInput,
              {
                backgroundColor: colors.surface,
                borderColor: colors.border,
                color: colors.text,
              }
            ]}
            placeholder="What's on your mind?"
            placeholderTextColor={colors.textSecondary}
            value={postContent}
            onChangeText={setPostContent}
            multiline
            numberOfLines={6}
            textAlignVertical="top"
          />
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  tabBar: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    paddingTop: 10,
  },
  tabItem: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 2,
    borderBottomColor: 'transparent',
  },
  tabLabel: {
    fontSize: 12,
    marginTop: 4,
    fontWeight: '500',
  },
  feedItem: {
    margin: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
  },
  feedHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    marginRight: 12,
  },
  feedAuthor: {
    flex: 1,
  },
  authorName: {
    fontSize: 16,
    fontWeight: '600',
  },
  feedTime: {
    fontSize: 12,
    marginTop: 2,
  },
  feedContent: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 12,
  },
  tradeData: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    marginBottom: 12,
  },
  tradeSymbol: {
    fontSize: 16,
    fontWeight: '600',
    flex: 1,
  },
  tradeSide: {
    fontSize: 14,
    fontWeight: '600',
    marginRight: 12,
  },
  tradePrice: {
    fontSize: 16,
    fontWeight: '600',
  },
  feedActions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  actionText: {
    fontSize: 12,
    marginLeft: 4,
  },
  traderItem: {
    margin: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
  },
  traderHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  traderAvatar: {
    width: 50,
    height: 50,
    borderRadius: 25,
    marginRight: 12,
  },
  traderInfo: {
    flex: 1,
  },
  traderNameRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  traderName: {
    fontSize: 16,
    fontWeight: '600',
    marginRight: 6,
  },
  traderTier: {
    fontSize: 12,
    marginTop: 2,
  },
  followButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
  },
  followButtonText: {
    fontSize: 14,
    fontWeight: '600',
  },
  traderStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 16,
    fontWeight: '600',
  },
  statLabel: {
    fontSize: 12,
    marginTop: 2,
  },
  strategyItem: {
    margin: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
  },
  strategyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  strategyName: {
    fontSize: 16,
    fontWeight: '600',
    flex: 1,
  },
  riskBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  riskText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#ffffff',
  },
  strategyDescription: {
    fontSize: 14,
    lineHeight: 18,
    marginBottom: 12,
  },
  strategyStats: {
    marginBottom: 16,
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  subscribeButton: {
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  subscribeButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  leaderboardItem: {
    flexDirection: 'row',
    alignItems: 'center',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
  },
  rankContainer: {
    width: 40,
    alignItems: 'center',
  },
  rank: {
    fontSize: 16,
    fontWeight: '600',
  },
  leaderboardAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    marginRight: 12,
  },
  leaderboardInfo: {
    flex: 1,
  },
  leaderboardName: {
    fontSize: 16,
    fontWeight: '600',
  },
  leaderboardTier: {
    fontSize: 12,
    marginTop: 2,
  },
  leaderboardValue: {
    alignItems: 'flex-end',
  },
  valueText: {
    fontSize: 16,
    fontWeight: '600',
  },
  changeText: {
    fontSize: 12,
    marginTop: 2,
  },
  fab: {
    position: 'absolute',
    bottom: 20,
    right: 20,
    width: 56,
    height: 56,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  modalContainer: {
    flex: 1,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
  },
  modalCancel: {
    fontSize: 16,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  modalPost: {
    fontSize: 16,
    fontWeight: '600',
  },
  postTypeSelector: {
    flexDirection: 'row',
    padding: 16,
    gap: 8,
  },
  postTypeButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
  },
  postTypeLabel: {
    fontSize: 12,
    marginLeft: 4,
    fontWeight: '500',
  },
  postInput: {
    margin: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    fontSize: 16,
    minHeight: 120,
  },
});

export default SocialScreen;