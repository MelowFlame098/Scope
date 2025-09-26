'use client';

import React, { useEffect } from 'react';
import {
  PlusIcon,
  StarIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { StarIcon as StarIconSolid } from '@heroicons/react/24/solid';
import { useStore } from '../store/useStore';
import { useAssetsData } from '../hooks';
import { formatCurrency, formatPercentage, getChangeColor } from '../utils';
import LoadingSpinner from './ui/LoadingSpinner';

const Watchlist: React.FC = () => {
  const {
    watchlist,
    selectedAssets,
    searchQuery,
    isLoading,
    addToWatchlist,
    removeFromWatchlist,
    addSelectedAsset,
    removeSelectedAsset,
    setSearchQuery
  } = useStore();
  
  const { subscribeToAssets } = useAssetsData();

  // Subscribe to real-time updates for watchlist assets
  useEffect(() => {
    if (watchlist?.length > 0) {
      const assetIds = (watchlist || []).map(asset => asset.id);
      subscribeToAssets(assetIds);
    }
  }, [watchlist, subscribeToAssets]);

  const filteredAssets = (watchlist || []).filter(asset =>
    asset.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
    asset.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const toggleFavorite = (assetId: string) => {
    const asset = watchlist?.find(a => a.id === assetId);
    if (asset) {
      if (asset.isFavorite) {
        removeFromWatchlist(assetId);
      } else {
        addToWatchlist({ ...asset, isFavorite: true });
      }
    }
  };

  const handleAssetSelect = (asset: any) => {
    const isSelected = selectedAssets.find(a => a.id === asset.id);
    if (isSelected) {
      removeSelectedAsset(asset.id);
    } else {
      addSelectedAsset(asset);
    }
  };

  const isAssetSelected = (assetId: string) => {
    return selectedAssets.some(asset => asset.id === assetId);
  };

  if (isLoading && (!watchlist || watchlist.length === 0)) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="md" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Watchlist
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {watchlist?.length || 0} assets tracked
          </p>
        </div>
        <button className="flex items-center space-x-2 px-3 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors text-sm">
          <PlusIcon className="h-4 w-4" />
          <span>Add Asset</span>
        </button>
      </div>

      {/* Search */}
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <MagnifyingGlassIcon className="h-4 w-4 text-gray-400" />
        </div>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="block w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg leading-5 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
          placeholder="Search watchlist..."
        />
      </div>

      {/* Selected Assets */}
      {selectedAssets.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Selected Assets ({selectedAssets.length})
          </h4>
          <div className="space-y-1">
            {selectedAssets.map((asset) => (
              <div
                key={`selected-${asset.id}`}
                className="flex items-center justify-between p-2 rounded-lg bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800"
              >
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                  <div>
                    <div className="font-medium text-primary-700 dark:text-primary-300 text-sm">
                      {asset.symbol}
                    </div>
                    <div className="text-xs text-primary-600 dark:text-primary-400">
                      {formatCurrency(asset.price)}
                    </div>
                  </div>
                </div>
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

      {/* Watchlist Assets */}
      <div className="space-y-1 max-h-96 overflow-y-auto">
        {filteredAssets.map((asset) => {
          const isSelected = isAssetSelected(asset.id);
          
          return (
            <div
              key={asset.id}
              className={`flex items-center justify-between p-3 rounded-lg border transition-all cursor-pointer ${
                isSelected
                  ? 'border-primary-200 dark:border-primary-800 bg-primary-50 dark:bg-primary-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50'
              }`}
              onClick={() => handleAssetSelect(asset)}
            >
              <div className="flex items-center space-x-3">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleFavorite(asset.id);
                  }}
                  className="text-gray-400 hover:text-yellow-500 transition-colors"
                >
                  {asset.isFavorite ? (
                    <StarIconSolid className="h-4 w-4 text-yellow-500" />
                  ) : (
                    <StarIcon className="h-4 w-4" />
                  )}
                </button>
                
                <div className="flex items-center space-x-2">
                  {/* Asset Icon/Category Indicator */}
                  <div className={`w-2 h-2 rounded-full ${
                    asset.category === 'crypto' ? 'bg-orange-400' :
                    asset.category === 'stocks' ? 'bg-blue-400' :
                    asset.category === 'forex' ? 'bg-green-400' :
                    'bg-purple-400'
                  }`}></div>
                  
                  <div>
                    <div className="font-medium text-gray-900 dark:text-white text-sm">
                      {asset.symbol}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 truncate max-w-24">
                      {asset.name}
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="text-right">
                <div className="font-medium text-gray-900 dark:text-white text-sm">
                  {formatCurrency(asset.price)}
                </div>
                <div className={`text-xs ${getChangeColor(asset.change24h)}`}>
                  {formatPercentage(asset.change24h)}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {filteredAssets.length === 0 && (
        <div className="text-center py-8">
          <div className="text-gray-500 dark:text-gray-400 text-sm">
            {searchQuery ? 'No assets found matching your search.' : 'Your watchlist is empty.'}
          </div>
          {!searchQuery && (
            <button className="mt-2 text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300 text-sm">
              Add your first asset
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default Watchlist;