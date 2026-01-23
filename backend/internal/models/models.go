package models

import (
	"time"

	"github.com/google/uuid"
	"github.com/shopspring/decimal"
	"gorm.io/gorm"
)

// Base model with UUID primary key
type Base struct {
	ID        uuid.UUID `gorm:"type:uuid;primary_key;default:gen_random_uuid()"`
	CreatedAt time.Time
	UpdatedAt time.Time
	DeletedAt gorm.DeletedAt `gorm:"index"`
}

type User struct {
	Base
	Email        string `gorm:"uniqueIndex;not null"`
	PasswordHash string `gorm:"not null"`
	FullName     string
	IsVerified   bool
	Role         string `gorm:"default:'user'"` // user, admin

	// Relations
	Portfolios []Portfolio
}

type Portfolio struct {
	Base
	UserID uuid.UUID `gorm:"index"`
	Name   string

	// Relations
	Items []PortfolioItem
}

type PortfolioItem struct {
	Base
	PortfolioID uuid.UUID       `gorm:"index"`
	StockSymbol string          `gorm:"index"` // e.g., AAPL
	Quantity    decimal.Decimal `gorm:"type:numeric"`
	AvgBuyPrice decimal.Decimal `gorm:"type:numeric"`
}

type Transaction struct {
	Base
	UserID      uuid.UUID       `gorm:"index"`
	Type        string          `gorm:"index"` // DEPOSIT, WITHDRAWAL, BUY, SELL
	Amount      decimal.Decimal `gorm:"type:numeric"`
	Currency    string          `gorm:"default:'USD'"`
	Status      string          `gorm:"default:'PENDING'"` // PENDING, COMPLETED, FAILED
	ReferenceID string          // External reference or Order ID
}

// Ensure UUID generation before creation
func (base *Base) BeforeCreate(tx *gorm.DB) (err error) {
	if base.ID == uuid.Nil {
		base.ID = uuid.New()
	}
	return
}
