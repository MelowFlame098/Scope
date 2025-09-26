'use client';

import React, { useEffect } from 'react';
import {
  MagnifyingGlassIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  GlobeAltIcon,
  BuildingLibraryIcon,
  XMarkIcon,
  StarIcon,
} from '@heroicons/react/24/outline';
import { CheckIcon, StarIcon as StarIconSolid } from '@heroicons/react/24/solid';
import { useStore } from '../store/useStore';
import { useAssetsData } from '../hooks';
import { formatCurrency, formatLargeNumber, formatPercentage, getAssetIcon } from '../utils';
import LoadingSpinner from './ui/LoadingSpinner';

const AssetSelector: React.FC = () => {
  const {
    assets,
    selectedAssets,
    watchlist,
    searchQuery,
    isLoading,
    addSelectedAsset,
    removeSelectedAsset,
    addToWatchlist,
    removeFromWatchlist,
    setSearchQuery,
    fetchAssets,
  } = useStore();
  
  const { subscribeToAssets } = useAssetsData();
  
  const [filter, setFilter] = React.useState<'all' | 'crypto' | 'stock' | 'forex' | 'commodity' | 'index'>('all');
  const [sortBy, setSortBy] = React.useState<'symbol' | 'price' | 'change' | 'volume' | 'marketCap'>('symbol');
  const [sortOrder, setSortOrder] = React.useState<'asc' | 'desc'>('asc');

  // Fetch assets on component mount
  useEffect(() => {
    fetchAssets();
  }, [fetchAssets]);

  // Subscribe to real-time asset data for selected assets
  useEffect(() => {
    if (selectedAssets.length > 0) {
      const assetIds = selectedAssets.map(asset => asset.id);
      subscribeToAssets(assetIds);
    }
  }, [selectedAssets, subscribeToAssets]);

  const filteredAssets = React.useMemo(() => {
    let filtered = assets.filter(asset => {
      const matchesSearch = asset.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           asset.name.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesFilter = filter === 'all' || asset.category === filter;
      return matchesSearch && matchesFilter;
    });

    // Sort assets
    filtered.sort((a, b) => {
      let aValue: any, bValue: any;
      
      switch (sortBy) {
        case 'symbol':
          aValue = a.symbol;
          bValue = b.symbol;
          break;
        case 'price':
          aValue = a.price;
          bValue = b.price;
          break;
        case 'change':
          aValue = a.change24h;
          bValue = b.change24h;
          break;
        case 'volume':
          aValue = a.volume || 0;
          bValue = b.volume || 0;
          break;
        case 'marketCap':
          aValue = a.marketCap || 0;
          bValue = b.marketCap || 0;
          break;
        default:
          return 0;
      }
      
      if (typeof aValue === 'string') {
        return sortOrder === 'asc' 
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }
      
      return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
    });

    return filtered;
  }, [assets, searchQuery, filter, sortBy, sortOrder]);

  const toggleAsset = (asset: any) => {
    const isSelected = selectedAssets.find(a => a.id === asset.id);
    if (isSelected) {
      removeSelectedAsset(asset.id);
    } else {
      addSelectedAsset(asset);
    }
  };

  const toggleWatchlist = (asset: any, e: React.MouseEvent) => {
    e.stopPropagation();
    const isInWatchlist = watchlist.some(item => item.id === asset.id);
    if (isInWatchlist) {
      removeFromWatchlist(asset.id);
    } else {
      addToWatchlist(asset);
    }
  };

  const handleSort = (newSortBy: typeof sortBy) => {
    if (sortBy === newSortBy) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(newSortBy);
      setSortOrder('asc');
    }
  };

  const getTypeColor = (category: string) => {
    switch (category) {
      case 'crypto':
        return 'bg-orange-100 dark:bg-orange-900/20 text-orange-700 dark:text-orange-400';
      case 'stock':
        return 'bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400';
      case 'forex':
        return 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400';
      case 'commodity':
        return 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-400';
      case 'index':
        return 'bg-purple-100 dark:bg-purple-900/20 text-purple-700 dark:text-purple-400';
      default:
        return 'bg-gray-100 dark:bg-gray-900/20 text-gray-700 dark:text-gray-400';
    }
  };

  const isAssetSelected = (assetId: string) => {
    return selectedAssets.some(asset => asset.id === assetId);
  };

  const isInWatchlist = (assetId: string) => {
    return watchlist.some(item => item.id === assetId);
  };

  const filters = [
    { value: 'all', label: 'All Assets' },
    { value: 'crypto', label: 'Crypto' },
    { value: 'stock', label: 'Stocks' },
    { value: 'forex', label: 'Forex' },
    { value: 'commodity', label: 'Commodities' },
    { value: 'index', label: 'Indices' },
  ];

  const sortOptions = [
    { value: 'symbol', label: 'Symbol' },
    { value: 'price', label: 'Price' },
    { value: 'change', label: 'Change' },
    { value: 'volume', label: 'Volume' },
    { value: 'marketCap', label: 'Market Cap' },
  ];

  if (isLoading && assets.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-center h-64">
          <LoadingSpinner size="lg" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Asset Selection
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {selectedAssets.length} assets selected • {filteredAssets.length} available
          </p>
        </div>
      </div>

      {/* Search and Controls */}
      <div className="space-y-3">
        {/* Search */}
        <div className="relative">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search assets..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Filter and Sort */}
        <div className="flex flex-wrap gap-2">
          {/* Filter Tabs */}
          <div className="flex space-x-1">
            {filters.map((filterOption) => (
              <button
                key={filterOption.value}
                onClick={() => setFilter(filterOption.value as any)}
                className={`px-3 py-1 text-sm font-medium rounded-lg whitespace-nowrap transition-colors ${
                  filter === filterOption.value
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                {filterOption.label}
              </button>
            ))}
          </div>

          {/* Sort */}
          <select
            value={`${sortBy}-${sortOrder}`}
            onChange={(e) => {
              const [newSortBy, newSortOrder] = e.target.value.split('-');
              setSortBy(newSortBy as any);
              setSortOrder(newSortOrder as any);
            }}
            className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            {sortOptions.map((option) => (
              <React.Fragment key={option.value}>
                <option value={`${option.value}-asc`}>{option.label} ↑</option>
                <option value={`${option.value}-desc`}>{option.label} ↓</option>
              </React.Fragment>
            ))}
          </select>
        </div>
      </div>

      {/* Selected Assets */}
      {selectedAssets.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Selected Assets ({selectedAssets.length})
          </h4>
          <div className="flex flex-wrap gap-2">
            {selectedAssets.map((asset) => (
              <div
                key={`selected-${asset.id}`}
                className="flex items-center space-x-2 px-3 py-2 bg-primary-100 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 rounded-lg border border-primary-200 dark:border-primary-800"
              >
                <div className="w-4 h-4">
                  {getAssetIcon(asset.category)}
                </div>
                <span className="font-medium text-sm">{asset.symbol}</span>
                <span className={`text-xs ${
                  asset.change24h >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {formatPercentage(asset.change24h)}
                </span>
                <button
                  onClick={() => removeSelectedAsset(asset.id)}
                  className="text-primary-500 hover:text-primary-700 dark:hover:text-primary-300 transition-colors"
                >
                  <XMarkIcon className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Assets List */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {filteredAssets.map((asset) => {
          const isSelected = isAssetSelected(asset.id);
          const inWatchlist = isInWatchlist(asset.id);
          
          return (
            <div
              key={asset.id}
              className={`p-3 rounded-lg border transition-all cursor-pointer ${
                isSelected
                  ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/10'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
              onClick={() => toggleAsset(asset)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-6 h-6">
                    {getAssetIcon(asset.category)}
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold text-gray-900 dark:text-white text-sm">
                        {asset.symbol}
                      </span>
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                        getTypeColor(asset.category)
                      }`}>
                        {asset.category.toUpperCase()}
                      </span>
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      {asset.name}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <div className="font-semibold text-gray-900 dark:text-white text-sm">
                      {formatCurrency(asset.price)}
                    </div>
                    <div className={`text-xs ${
                      asset.change24h >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                    }`}>
                      {formatPercentage(asset.change24h)}
                    </div>
                  </div>
                  
                  {asset.volume && (
                    <div className="text-right text-xs text-gray-500 dark:text-gray-400">
                      <div>Vol: {formatLargeNumber(asset.volume)}</div>
                {asset.marketCap && (
                <div>MCap: {formatLargeNumber(asset.marketCap)}</div>
                      )}
                    </div>
                  )}
                  
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={(e) => toggleWatchlist(asset, e)}
                      className={`transition-colors ${
                        inWatchlist
                          ? 'text-yellow-500 hover:text-yellow-600'
                          : 'text-gray-400 hover:text-yellow-500'
                      }`}
                      title={inWatchlist ? 'Remove from watchlist' : 'Add to watchlist'}
                    >
                      {inWatchlist ? (
                        <StarIconSolid className="h-4 w-4" />
                      ) : (
                        <StarIcon className="h-4 w-4" />
                      )}
                    </button>
                    
                    {isSelected && (
                      <div className="w-5 h-5 bg-primary-600 rounded-full flex items-center justify-center">
                        <CheckIcon className="h-3 w-3 text-white" />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {filteredAssets.length === 0 && (
        <div className="text-center py-8">
          <ChartBarIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500 dark:text-gray-400">
            {searchQuery ? 'No assets found matching your search.' : 'No assets available.'}
          </p>
        </div>
      )}
    </div>
  );
};

export default AssetSelector;