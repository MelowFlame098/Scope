'use client'

import { useState, useEffect } from 'react'
import {
  ChatBubbleLeftRightIcon,
  UserGroupIcon,
  FireIcon,
  ClockIcon,
  HandThumbUpIcon,
  HandThumbDownIcon,
  ChatBubbleOvalLeftIcon,
  PlusIcon,
  FunnelIcon,
  MagnifyingGlassIcon,
  TagIcon,
  ArrowUpIcon,
  ArrowDownIcon
} from '@heroicons/react/24/outline'
import {
  HandThumbUpIcon as HandThumbUpSolidIcon,
  HandThumbDownIcon as HandThumbDownSolidIcon
} from '@heroicons/react/24/solid'

interface ForumPost {
  id: string
  title: string
  content: string
  author: {
    name: string
    avatar: string
    reputation: number
    verified: boolean
  }
  category: 'crypto' | 'stocks' | 'forex' | 'general' | 'analysis'
  tags: string[]
  createdAt: string
  updatedAt: string
  likes: number
  dislikes: number
  comments: number
  views: number
  trending: boolean
  userVote?: 'up' | 'down' | null
}

interface Comment {
  id: string
  postId: string
  content: string
  author: {
    name: string
    avatar: string
    reputation: number
  }
  createdAt: string
  likes: number
  userVote?: 'up' | 'down' | null
}

const mockPosts: ForumPost[] = [
  {
    id: '1',
    title: 'Bitcoin Breaking $45K - Is This the Start of the Next Bull Run?',
    content: 'Bitcoin has successfully broken through the $45,000 resistance level with significant volume. Looking at the technical indicators, we\'re seeing strong momentum with RSI at 68 and MACD showing positive divergence. What are your thoughts on the next price targets?',
    author: {
      name: 'CryptoAnalyst_Pro',
      avatar: '/api/placeholder/40/40',
      reputation: 2847,
      verified: true
    },
    category: 'crypto',
    tags: ['bitcoin', 'technical-analysis', 'bullish'],
    createdAt: '2024-01-15T10:30:00Z',
    updatedAt: '2024-01-15T10:30:00Z',
    likes: 156,
    dislikes: 23,
    comments: 47,
    views: 1234,
    trending: true,
    userVote: null
  },
  {
    id: '2',
    title: 'Tesla Q4 Earnings Preview - What to Expect',
    content: 'Tesla is set to report Q4 earnings next week. Key metrics to watch: delivery numbers, margin compression, and guidance for 2024. The stock has been volatile lately, trading between $240-$270. Here\'s my analysis of what could move the stock...',
    author: {
      name: 'StockGuru2024',
      avatar: '/api/placeholder/40/40',
      reputation: 1523,
      verified: false
    },
    category: 'stocks',
    tags: ['tesla', 'earnings', 'analysis'],
    createdAt: '2024-01-15T09:15:00Z',
    updatedAt: '2024-01-15T09:15:00Z',
    likes: 89,
    dislikes: 12,
    comments: 28,
    views: 567,
    trending: false,
    userVote: 'up'
  },
  {
    id: '3',
    title: 'Fed Rate Decision Impact on Forex Markets',
    content: 'With the Fed meeting approaching, how do you think different rate scenarios will affect major currency pairs? EUR/USD has been consolidating around 1.0875, but I expect significant movement post-announcement.',
    author: {
      name: 'ForexTrader_Elite',
      avatar: '/api/placeholder/40/40',
      reputation: 3156,
      verified: true
    },
    category: 'forex',
    tags: ['fed', 'interest-rates', 'eurusd'],
    createdAt: '2024-01-15T08:45:00Z',
    updatedAt: '2024-01-15T08:45:00Z',
    likes: 67,
    dislikes: 8,
    comments: 19,
    views: 423,
    trending: false,
    userVote: null
  },
  {
    id: '4',
    title: 'Portfolio Diversification Strategy for 2024',
    content: 'Looking to rebalance my portfolio for the new year. Currently 60% stocks, 25% crypto, 10% bonds, 5% commodities. Considering increasing crypto allocation to 35%. Thoughts on this allocation given current market conditions?',
    author: {
      name: 'InvestorMike',
      avatar: '/api/placeholder/40/40',
      reputation: 892,
      verified: false
    },
    category: 'general',
    tags: ['portfolio', 'diversification', 'strategy'],
    createdAt: '2024-01-15T07:20:00Z',
    updatedAt: '2024-01-15T07:20:00Z',
    likes: 45,
    dislikes: 5,
    comments: 31,
    views: 289,
    trending: false,
    userVote: null
  }
]

const mockComments: Comment[] = [
  {
    id: '1',
    postId: '1',
    content: 'Great analysis! I agree that $50K is the next major resistance. The institutional buying pressure is definitely supporting this move.',
    author: {
      name: 'BTCMaximalist',
      avatar: '/api/placeholder/32/32',
      reputation: 1245
    },
    createdAt: '2024-01-15T10:45:00Z',
    likes: 23,
    userVote: null
  },
  {
    id: '2',
    postId: '1',
    content: 'Not so fast. We need to see sustained volume above $45K. Could be a bull trap if we don\'t hold this level.',
    author: {
      name: 'BearishBob',
      avatar: '/api/placeholder/32/32',
      reputation: 567
    },
    createdAt: '2024-01-15T11:00:00Z',
    likes: 12,
    userVote: 'up'
  }
]

const categories = {
  all: { name: 'All Posts', icon: ChatBubbleLeftRightIcon },
  crypto: { name: 'Crypto', icon: FireIcon },
  stocks: { name: 'Stocks', icon: ArrowUpIcon },
    forex: { name: 'Forex', icon: ArrowDownIcon },
  analysis: { name: 'Analysis', icon: ChatBubbleOvalLeftIcon },
  general: { name: 'General', icon: UserGroupIcon }
}

const sortOptions = [
  { value: 'trending', label: 'Trending' },
  { value: 'newest', label: 'Newest' },
  { value: 'popular', label: 'Most Popular' },
  { value: 'discussed', label: 'Most Discussed' }
]

export default function CommunityForum() {
  const [posts, setPosts] = useState<ForumPost[]>(mockPosts)
  const [comments, setComments] = useState<Comment[]>(mockComments)
  const [activeCategory, setActiveCategory] = useState<string>('all')
  const [sortBy, setSortBy] = useState<string>('trending')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedPost, setSelectedPost] = useState<ForumPost | null>(null)
  const [showNewPostModal, setShowNewPostModal] = useState(false)
  const [newComment, setNewComment] = useState('')
  const [isClient, setIsClient] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  const filteredPosts = posts.filter(post => {
    if (activeCategory !== 'all' && post.category !== activeCategory) return false
    if (searchQuery && !post.title.toLowerCase().includes(searchQuery.toLowerCase()) && 
        !post.content.toLowerCase().includes(searchQuery.toLowerCase())) return false
    return true
  })

  const sortedPosts = [...filteredPosts].sort((a, b) => {
    switch (sortBy) {
      case 'trending':
        return (b.trending ? 1 : 0) - (a.trending ? 1 : 0) || b.likes - a.likes
      case 'newest':
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      case 'popular':
        return b.likes - a.likes
      case 'discussed':
        return b.comments - a.comments
      default:
        return 0
    }
  })

  const handleVote = (postId: string, voteType: 'up' | 'down') => {
    setPosts(prev => prev.map(post => {
      if (post.id === postId) {
        const currentVote = post.userVote
        let newLikes = post.likes
        let newDislikes = post.dislikes
        let newUserVote: 'up' | 'down' | null = voteType

        // Remove previous vote
        if (currentVote === 'up') newLikes--
        if (currentVote === 'down') newDislikes--

        // Add new vote or remove if same
        if (currentVote === voteType) {
          newUserVote = null
        } else {
          if (voteType === 'up') newLikes++
          if (voteType === 'down') newDislikes++
        }

        return { ...post, likes: newLikes, dislikes: newDislikes, userVote: newUserVote }
      }
      return post
    }))
  }

  const handleCommentVote = (commentId: string, voteType: 'up' | 'down') => {
    setComments(prev => prev.map(comment => {
      if (comment.id === commentId) {
        const currentVote = comment.userVote
        let newLikes = comment.likes
        let newUserVote: 'up' | 'down' | null = voteType

        if (currentVote === 'up') newLikes--
        if (currentVote === voteType) {
          newUserVote = null
        } else {
          if (voteType === 'up') newLikes++
        }

        return { ...comment, likes: newLikes, userVote: newUserVote }
      }
      return comment
    }))
  }

  const formatTimestamp = (timestamp: string): string => {
    if (!isClient) {
      return ''
    }
    
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    return date.toLocaleDateString()
  }

  const postComments = comments.filter(comment => comment.postId === selectedPost?.id)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white flex items-center">
            <UserGroupIcon className="h-6 w-6 mr-2" />
            Community Forum
          </h2>
          
          <button
            onClick={() => setShowNewPostModal(true)}
            className="btn-primary flex items-center space-x-2"
          >
            <PlusIcon className="h-4 w-4" />
            <span>New Post</span>
          </button>
        </div>

        {/* Search and Filters */}
        <div className="space-y-4">
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-dark-400" />
            <input
              type="text"
              placeholder="Search discussions..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input-field pl-10"
            />
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Category Filter */}
            <div className="flex space-x-1">
              {Object.entries(categories).map(([key, cat]) => {
                const IconComponent = cat.icon
                return (
                  <button
                    key={key}
                    onClick={() => setActiveCategory(key)}
                    className={`flex items-center space-x-1 px-3 py-1 text-sm rounded transition-colors ${
                      activeCategory === key
                        ? 'bg-primary-600 text-white'
                        : 'bg-dark-700 text-dark-300 hover:bg-dark-600'
                    }`}
                  >
                    <IconComponent className="h-4 w-4" />
                    <span>{cat.name}</span>
                  </button>
                )
              })}
            </div>

            {/* Sort Filter */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-dark-700 border border-dark-600 text-white text-sm rounded px-2 py-1"
            >
              {sortOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Posts List */}
        <div className="lg:col-span-2 space-y-4">
          {sortedPosts.map(post => {
            const isSelected = selectedPost?.id === post.id
            
            return (
              <div
                key={post.id}
                onClick={() => setSelectedPost(post)}
                className={`card cursor-pointer transition-all duration-200 ${
                  isSelected ? 'ring-2 ring-primary-500' : 'hover:bg-dark-750'
                }`}
              >
                <div className="flex items-start space-x-4">
                  {/* Author Avatar */}
                  <div className="w-10 h-10 bg-primary-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white font-medium">
                      {post.author.name.charAt(0)}
                    </span>
                  </div>

                  {/* Post Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <h3 className="text-white font-semibold line-clamp-1">
                            {post.title}
                          </h3>
                          {post.trending && (
                            <FireIcon className="h-4 w-4 text-orange-400" />
                          )}
                        </div>
                        
                        <p className="text-dark-300 text-sm mb-2 line-clamp-2">
                          {post.content}
                        </p>
                        
                        <div className="flex items-center space-x-4 text-xs text-dark-400">
                          <div className="flex items-center space-x-1">
                            <span>{post.author.name}</span>
                            {post.author.verified && (
                              <span className="text-blue-400">✓</span>
                            )}
                            <span>({post.author.reputation} rep)</span>
                          </div>
                          
                          <div className="flex items-center space-x-1">
                            <ClockIcon className="h-3 w-3" />
                            <span>{formatTimestamp(post.createdAt)}</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Tags */}
                    <div className="flex flex-wrap gap-1 mt-2">
                      {post.tags.slice(0, 3).map(tag => (
                        <span
                          key={tag}
                          className="px-2 py-1 bg-dark-600 text-dark-300 text-xs rounded"
                        >
                          #{tag}
                        </span>
                      ))}
                    </div>

                    {/* Post Stats */}
                    <div className="flex items-center justify-between mt-3">
                      <div className="flex items-center space-x-4">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleVote(post.id, 'up')
                          }}
                          className={`flex items-center space-x-1 text-sm ${
                            post.userVote === 'up' ? 'text-green-400' : 'text-dark-400 hover:text-green-400'
                          } transition-colors`}
                        >
                          {post.userVote === 'up' ? (
                            <HandThumbUpSolidIcon className="h-4 w-4" />
                          ) : (
                            <HandThumbUpIcon className="h-4 w-4" />
                          )}
                          <span>{post.likes}</span>
                        </button>
                        
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleVote(post.id, 'down')
                          }}
                          className={`flex items-center space-x-1 text-sm ${
                            post.userVote === 'down' ? 'text-red-400' : 'text-dark-400 hover:text-red-400'
                          } transition-colors`}
                        >
                          {post.userVote === 'down' ? (
                            <HandThumbDownSolidIcon className="h-4 w-4" />
                          ) : (
                            <HandThumbDownIcon className="h-4 w-4" />
                          )}
                          <span>{post.dislikes}</span>
                        </button>
                        
                        <div className="flex items-center space-x-1 text-sm text-dark-400">
                          <ChatBubbleOvalLeftIcon className="h-4 w-4" />
                          <span>{post.comments}</span>
                        </div>
                      </div>
                      
                      <span className="text-xs text-dark-500">
                        {post.views} views
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        {/* Post Details / Comments */}
        <div className="lg:col-span-1">
          {selectedPost ? (
            <div className="space-y-4">
              {/* Selected Post Details */}
              <div className="card">
                <h3 className="text-lg font-semibold text-white mb-2">
                  {selectedPost.title}
                </h3>
                
                <p className="text-dark-300 text-sm leading-relaxed mb-4">
                  {selectedPost.content}
                </p>
                
                <div className="flex flex-wrap gap-1 mb-4">
                  {selectedPost.tags.map(tag => (
                    <span
                      key={tag}
                      className="px-2 py-1 bg-primary-900/30 text-primary-400 text-xs rounded"
                    >
                      #{tag}
                    </span>
                  ))}
                </div>
                
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center space-x-2">
                    <div className="w-6 h-6 bg-primary-600 rounded-full flex items-center justify-center">
                      <span className="text-white text-xs">
                        {selectedPost.author.name.charAt(0)}
                      </span>
                    </div>
                    <span className="text-white">{selectedPost.author.name}</span>
                    {selectedPost.author.verified && (
                      <span className="text-blue-400">✓</span>
                    )}
                  </div>
                  
                  <span className="text-dark-400">
                    {formatTimestamp(selectedPost.createdAt)}
                  </span>
                </div>
              </div>

              {/* Comments */}
              <div className="card">
                <h4 className="text-lg font-semibold text-white mb-4">
                  Comments ({postComments.length})
                </h4>
                
                {/* New Comment */}
                <div className="mb-4">
                  <textarea
                    value={newComment}
                    onChange={(e) => setNewComment(e.target.value)}
                    placeholder="Share your thoughts..."
                    className="input-field h-20 resize-none"
                  />
                  <button
                    onClick={() => {
                      if (newComment.trim()) {
                        // Add comment logic here
                        setNewComment('')
                      }
                    }}
                    className="btn-primary mt-2 text-sm"
                  >
                    Post Comment
                  </button>
                </div>
                
                {/* Comments List */}
                <div className="space-y-3 max-h-64 overflow-y-auto">
                  {postComments.map(comment => (
                    <div key={comment.id} className="p-3 bg-dark-700 rounded">
                      <div className="flex items-start space-x-2">
                        <div className="w-6 h-6 bg-dark-600 rounded-full flex items-center justify-center flex-shrink-0">
                          <span className="text-white text-xs">
                            {comment.author.name.charAt(0)}
                          </span>
                        </div>
                        
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-1">
                            <span className="text-white text-sm font-medium">
                              {comment.author.name}
                            </span>
                            <span className="text-dark-400 text-xs">
                              ({comment.author.reputation} rep)
                            </span>
                            <span className="text-dark-500 text-xs">
                              {formatTimestamp(comment.createdAt)}
                            </span>
                          </div>
                          
                          <p className="text-dark-300 text-sm mb-2">
                            {comment.content}
                          </p>
                          
                          <button
                            onClick={() => handleCommentVote(comment.id, 'up')}
                            className={`flex items-center space-x-1 text-xs ${
                              comment.userVote === 'up' ? 'text-green-400' : 'text-dark-400 hover:text-green-400'
                            } transition-colors`}
                          >
                            {comment.userVote === 'up' ? (
                              <HandThumbUpSolidIcon className="h-3 w-3" />
                            ) : (
                              <HandThumbUpIcon className="h-3 w-3" />
                            )}
                            <span>{comment.likes}</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="card text-center py-12">
              <ChatBubbleLeftRightIcon className="h-16 w-16 text-dark-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-white mb-2">Select a Post</h3>
              <p className="text-dark-400">Choose a discussion to view details and comments</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}