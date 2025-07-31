const express = require('express');
const cors = require('cors');
const multer = require('multer');
const csv = require('csv-parser');
const fs = require('fs');
const path = require('path');
const natural = require('natural');
const stopword = require('stopword');
const sentiment = require('sentiment');
// Removed kmeans - using LDA topic modeling only
const _ = require('lodash');
const Anthropic = require('@anthropic-ai/sdk');

// Initialize Anthropic client (requires ANTHROPIC_API_KEY environment variable)
const anthropic = process.env.ANTHROPIC_API_KEY ? new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY
}) : null;

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

const upload = multer({ dest: 'uploads/' });

app.post('/api/analyze', upload.single('file'), async (req, res) => {
  const startTime = Date.now();
  const filePath = req.file.path;
  const results = [];
  
  fs.createReadStream(filePath)
    .pipe(csv())
    .on('data', (data) => results.push(data))
    .on('end', async () => {
      try {
        console.log(`Processing ${results.length} rows from CSV`);
        
        // Auto-detect comment columns (look for text-heavy columns)
        const sampleRow = results[0];
        if (!sampleRow) {
          throw new Error('CSV file appears to be empty or invalid');
        }
        
        console.log('CSV columns found:', Object.keys(sampleRow));
        
        const commentColumns = Object.keys(sampleRow).filter(key => {
          const avgLength = results.slice(0, 10).reduce((sum, row) => 
            sum + (row[key] || '').length, 0) / 10;
          return avgLength > 10; // Lowered from 20 to 10 chars - more lenient
        });
        
        // Fallback: if no comment columns detected, use all text columns
        if (commentColumns.length === 0) {
          console.log('No comment columns detected, using all columns with text data');
          const allColumns = Object.keys(sampleRow).filter(key => {
            const hasText = results.slice(0, 5).some(row => 
              (row[key] || '').trim().length > 0 && isNaN(row[key])
            );
            return hasText;
          });
          commentColumns.push(...allColumns);
        }
        
        console.log('Detected comment columns:', commentColumns);
        
        const comments = results.map(row => {
          const commentText = commentColumns.map(col => row[col] || '').join(' ');
          return commentText.toLowerCase().trim();
        }).filter(comment => comment.length > 3); // Lowered from 10 to 3 chars - more lenient
        
        console.log(`Extracted ${comments.length} valid comments from ${results.length} rows`);
        
        // Token estimation and warnings
        const avgTokensPerComment = 20; // Conservative estimate
        const estimatedTokens = comments.length * avgTokensPerComment;
        const maxTokensPerRequest = 8000; // Claude Haiku limit
        
        console.log(`Estimated tokens needed: ${estimatedTokens}`);
        
        if (estimatedTokens > 50000) {
          throw new Error(`Dataset too large: ${comments.length} comments (estimated ${estimatedTokens} tokens). Please reduce to under 2,000 comments to avoid API limits.`);
        }
        
        if (estimatedTokens > 25000) {
          console.warn(`Large dataset warning: ${comments.length} comments may take 2-3 minutes and consume significant API credits.`);
        }
        
        // Enhanced text processing
        const processedComments = comments.map((comment, index) => {
          // Use simple regex tokenization instead of Natural.js
          const tokens = comment.match(/\b[a-zA-Z]{3,}\b/g) || [];
          const filtered = stopword.removeStopwords(tokens)
            .filter(word => word.length > 2 && /^[a-zA-Z]+$/.test(word));
          return {
            originalIndex: index,
            originalText: comment,
            processedText: filtered.join(' '),
            wordCount: comment.split(/\s+/).length,
            tokens: filtered
          };
        }).filter(item => item.tokens.length > 0); // Lowered from 2 to 0 - more lenient
        
        console.log(`Processed ${processedComments.length} comments with sufficient tokens`);
        
        if (processedComments.length < 1) {
          throw new Error(`Not enough valid comments to analyze. Found ${processedComments.length} processable comments, need at least 1.`);
        }
        
        // LLM-First Theme Classification Approach
        console.log('Starting LLM-first theme classification...');
        let finalTopics = [];
        let coherenceScore = 0.8; // Default for LLM-based classification
        
        try {
          // Step 1: Have Claude identify themes from ALL comments
          console.log('Step 1: Having Claude identify themes from all comments...');
          
          // Check if Anthropic API is configured
          if (!anthropic) {
            console.warn('ANTHROPIC_API_KEY not set, using fallback theme identification');
            throw new Error('Anthropic API not configured - using fallback themes');
          }
          
          // Sample comments for theme identification (limit based on token estimate)
          const maxSampleSize = estimatedTokens > 25000 ? 30 : 50;
          const sampleComments = comments.slice(0, Math.min(maxSampleSize, comments.length));
          const commentsSample = sampleComments.map((comment, index) => 
            `${index + 1}. ${comment}`).join('\n');

          const themeIdentificationPrompt = `Analyze these comments and identify 3-8 distinct themes/categories that emerge from the content.

Comments:
${commentsSample}

Based on these comments, identify the main themes that appear. For each theme, provide:
1. Theme name (2-4 words, business-focused)
2. Brief description
3. Key indicators/words that signal this theme

Respond in JSON format with an array of themes:
{
  "themes": [
    {
      "name": "Theme Name",
      "description": "What this theme represents",
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }
  ]
}`;

          const themeResponse = await anthropic.messages.create({
            model: "claude-3-haiku-20240307",
            max_tokens: 1000,
            temperature: 0.3,
            messages: [{ role: "user", content: themeIdentificationPrompt }]
          });

          const themeData = JSON.parse(themeResponse.content[0].text);
          const identifiedThemes = themeData.themes;
          
          console.log(`Identified ${identifiedThemes.themes?.length || identifiedThemes.length} themes:`, identifiedThemes.map(t => t.name));

          // Step 2: Classify each comment into identified themes
          console.log('Step 2: Classifying each comment into themes...');
          
          const commentClassifications = [];
          // AGGRESSIVE batch sizing to minimize API calls - target max 10 total batches
          const maxBatches = 10;
          const batchSize = Math.max(50, Math.ceil(comments.length / maxBatches)); // Minimum 50 per batch
          const actualBatches = Math.ceil(comments.length / batchSize);
          
          console.log(`AGGRESSIVE BATCHING: ${comments.length} comments in ${actualBatches} batches of ~${batchSize} each`);
          console.log(`This will reduce API calls from 77 to ${actualBatches}`);
          
          for (let i = 0; i < comments.length; i += batchSize) {
            const batch = comments.slice(i, i + batchSize);
            const batchComments = batch.map((comment, index) => 
              `${i + index + 1}. ${comment}`).join('\n');

            // LONG delays to avoid rate limits completely
            if (i > 0) {
              console.log(`Waiting 10 seconds before batch ${Math.floor(i/batchSize) + 1}/${actualBatches} to avoid rate limits...`);
              await new Promise(resolve => setTimeout(resolve, 10000)); // 10 second delay
            }

            const classificationPrompt = `Classify each of these comments into one of the identified themes. Each comment should be assigned to exactly one theme.

Available themes:
${identifiedThemes.map((theme, idx) => 
  `${idx + 1}. ${theme.name}: ${theme.description}`).join('\n')}

Comments to classify:
${batchComments}

Respond in JSON format with an array of classifications:
{
  "classifications": [
    {
      "commentIndex": 1,
      "themeName": "Exact theme name from above",
      "confidence": 0.9
    }
  ]
}`;

            try {
              const classificationResponse = await anthropic.messages.create({
                model: "claude-3-haiku-20240307",
                max_tokens: 1500,
                temperature: 0.1,
                messages: [{ role: "user", content: classificationPrompt }]
              });

              let classificationData;
              try {
                classificationData = JSON.parse(classificationResponse.content[0].text);
              } catch (jsonError) {
                console.warn(`JSON parsing failed for batch starting at ${i}:`, jsonError.message);
                console.warn('Raw response:', classificationResponse.content[0].text);
                // Create fallback classifications
                classificationData = {
                  classifications: batch.map((comment, batchIndex) => ({
                    commentIndex: i + batchIndex + 1,
                    themeName: identifiedThemes[0]?.name || 'Uncategorized',
                    confidence: 0.5
                  }))
                };
              }
              
              commentClassifications.push(...classificationData.classifications);
              
              console.log(`Classified batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(comments.length/batchSize)}`);
            } catch (batchError) {
              console.warn(`Classification failed for batch starting at ${i}:`, batchError.message);
              
              // Handle rate limit errors specifically
              if (batchError.message.includes('429') || batchError.message.includes('rate_limit_error')) {
                console.warn('Rate limit hit, waiting 60 seconds before continuing...');
                await new Promise(resolve => setTimeout(resolve, 60000));
                // Retry this batch once
                try {
                  const retryResponse = await anthropic.messages.create({
                    model: "claude-3-haiku-20240307",
                    max_tokens: 1500,
                    temperature: 0.1,
                    messages: [{ role: "user", content: classificationPrompt }]
                  });
                  const retryData = JSON.parse(retryResponse.content[0].text);
                  commentClassifications.push(...retryData.classifications);
                  console.log(`Retry successful for batch ${Math.floor(i/batchSize) + 1}`);
                } catch (retryError) {
                  console.warn('Retry also failed, using fallback classifications');
                  batch.forEach((comment, batchIndex) => {
                    commentClassifications.push({
                      commentIndex: i + batchIndex + 1,
                      themeName: identifiedThemes[0]?.name || 'Uncategorized',
                      confidence: 0.5
                    });
                  });
                }
              } else {
                // Add fallback classifications for this batch
                batch.forEach((comment, batchIndex) => {
                  commentClassifications.push({
                    commentIndex: i + batchIndex + 1,
                    themeName: identifiedThemes[0]?.name || 'Uncategorized',
                    confidence: 0.5
                  });
                });
              }
            }
          }

          console.log(`Classified ${commentClassifications.length} comments into themes`);

          // Step 3: Group comments by theme and calculate accurate percentages
          console.log('Step 3: Grouping comments and calculating percentages...');
          
          const themeGroups = {};
          identifiedThemes.forEach(theme => {
            themeGroups[theme.name] = {
              name: theme.name,
              description: theme.description,
              keywords: theme.keywords,
              comments: [],
              commentIndices: []
            };
          });

          // Add uncategorized theme
          themeGroups['Uncategorized'] = {
            name: 'Uncategorized',
            description: 'Comments that could not be clearly categorized',
            keywords: [],
            comments: [],
            commentIndices: []
          };

          // Group comments by their assigned themes
          commentClassifications.forEach(classification => {
            const commentIndex = classification.commentIndex - 1; // Convert to 0-based
            const comment = comments[commentIndex];
            const themeName = classification.themeName || 'Uncategorized';
            
            if (themeGroups[themeName]) {
              themeGroups[themeName].comments.push({
                text: comment,
                originalIndex: commentIndex,
                confidence: classification.confidence || 0.5
              });
              themeGroups[themeName].commentIndices.push(commentIndex);
            }
          });

          // Build final topic analysis with accurate counts and percentages
          const themeGroupsArray = Object.values(themeGroups).filter(group => group.comments.length > 0);
          
          finalTopics = await Promise.all(themeGroupsArray.map(async (group, index) => {
              // SIMPLIFIED sentiment analysis - use local sentiment library instead of Claude
              console.log(`Analyzing sentiment for theme: ${group.name} using local analysis (avoiding API calls)`);
              let themeSentiments = [];
              
              try {
                // Use local sentiment analysis to avoid API rate limits entirely
                group.comments.forEach(comment => {
                  const score = sentiment(comment.text);
                  let classification = 'neutral';
                  
                  // Business context rules
                  const text = comment.text.toLowerCase();
                  if (text.includes('expensive') || text.includes('costly') || text.includes('overpriced') || 
                      text.includes('disappointed') || text.includes('terrible') || text.includes('awful') ||
                      text.includes('bad') || text.includes('worst') || text.includes('hate')) {
                    classification = 'negative';
                  } else if (text.includes('great') || text.includes('excellent') || text.includes('amazing') ||
                             text.includes('love') || text.includes('perfect') || text.includes('wonderful') ||
                             text.includes('best') || text.includes('fantastic')) {
                    classification = 'positive';
                  } else if (score.comparative > 0.1) {
                    classification = 'positive';
                  } else if (score.comparative < -0.1) {
                    classification = 'negative';
                  }
                  
                  themeSentiments.push({
                    classification: classification,
                    reasoning: `Local analysis: score ${score.comparative}`,
                    confidence: 0.8
                  });
                });
              } catch (error) {
                console.warn(`Theme sentiment analysis failed for ${group.name}:`, error.message);
                group.comments.forEach(() => {
                  themeSentiments.push({
                    classification: 'neutral',
                    reasoning: 'error fallback',
                    confidence: 0.5
                  });
                });
              }

              // Calculate sentiment distribution
              const sentimentCounts = {
                positive: themeSentiments.filter(s => s.classification === 'positive').length,
                negative: themeSentiments.filter(s => s.classification === 'negative').length,
                neutral: themeSentiments.filter(s => s.classification === 'neutral').length
              };
              
              // Overall theme sentiment based on majority
              const sentimentClassification = sentimentCounts.positive > sentimentCounts.negative && sentimentCounts.positive > sentimentCounts.neutral ? 'positive' :
                sentimentCounts.negative > sentimentCounts.positive && sentimentCounts.negative > sentimentCounts.neutral ? 'negative' : 'neutral';
              
              // Calculate average sentiment score for compatibility
              const avgSentiment = sentimentCounts.positive > 0 ? 0.3 : sentimentCounts.negative > 0 ? -0.3 : 0;

              const volume = group.comments.length;
              const percentage = Math.round((volume / comments.length) * 100);

              return {
                topicId: index + 1,
                title: group.name,
                llmDescription: group.description,
                words: group.keywords.map(keyword => ({ term: keyword, weight: 1, probability: 1 })),
                volume: volume,
                percentage: percentage,
                sentiment: {
                  classification: sentimentClassification,
                  score: Math.round(avgSentiment * 100) / 100,
                  distribution: {
                    positive: sentimentCounts.positive,
                    negative: sentimentCounts.negative,
                    neutral: sentimentCounts.neutral,
                    positivePercentage: Math.round((sentimentCounts.positive / volume) * 100),
                    negativePercentage: Math.round((sentimentCounts.negative / volume) * 100),
                    neutralPercentage: Math.round((sentimentCounts.neutral / volume) * 100)
                  }
                },
                confidence: Math.round(group.comments.reduce((sum, c) => sum + c.confidence, 0) / group.comments.length * 100),
                avgWordCount: Math.round(group.comments.reduce((sum, c) => 
                  sum + c.text.split(/\s+/).length, 0) / group.comments.length),
                businessImpact: sentimentClassification === 'negative' && percentage > 10 ? 'high' : 
                  percentage > 15 ? 'medium' : 'low',
                comments: group.comments, // ALL comments, not samples
                enhancedByAI: true
              };
            }));

          console.log('LLM-first theme classification completed');
          
        } catch (error) {
          console.error('LLM theme classification failed:', error.message);
          console.log('Using fallback keyword-based theme identification...');
          
          // Fallback: Use keyword-based theme identification
          const fallbackThemes = [
            { name: 'Value and Pricing', keywords: ['value', 'price', 'cost', 'expensive', 'cheap', 'worth', 'money', 'budget'] },
            { name: 'Service Quality', keywords: ['service', 'staff', 'help', 'friendly', 'rude', 'professional', 'assistance'] },
            { name: 'Amenities and Features', keywords: ['pool', 'gym', 'spa', 'restaurant', 'room', 'facility', 'amenity', 'activity'] },
            { name: 'Experience and Satisfaction', keywords: ['experience', 'enjoy', 'satisfied', 'disappointed', 'amazing', 'terrible', 'love', 'hate'] },
            { name: 'General Feedback', keywords: ['overall', 'general', 'think', 'feel', 'opinion', 'recommend', 'suggest'] }
          ];
          
          // Classify comments into fallback themes using keyword matching
          const themeGroups = {};
          fallbackThemes.forEach(theme => {
            themeGroups[theme.name] = {
              name: theme.name,
              description: `Comments related to ${theme.name.toLowerCase()}`,
              keywords: theme.keywords,
              comments: [],
              commentIndices: []
            };
          });
          
          // Add uncategorized theme
          themeGroups['Other'] = {
            name: 'Other',
            description: 'Comments that do not fit into specific categories',
            keywords: [],
            comments: [],
            commentIndices: []
          };
          
          // Classify each comment based on keyword matching
          comments.forEach((comment, index) => {
            const lowerComment = comment.toLowerCase();
            let bestMatch = 'Other';
            let maxMatches = 0;
            
            fallbackThemes.forEach(theme => {
              const matches = theme.keywords.filter(keyword => 
                lowerComment.includes(keyword.toLowerCase())).length;
              if (matches > maxMatches) {
                maxMatches = matches;
                bestMatch = theme.name;
              }
            });
            
            themeGroups[bestMatch].comments.push({
              text: comment,
              originalIndex: index,
              confidence: maxMatches > 0 ? 0.7 : 0.3
            });
            themeGroups[bestMatch].commentIndices.push(index);
          });
          
          // Build final topics from fallback themes
          const themeGroupsArray = Object.values(themeGroups).filter(group => group.comments.length > 0);
          
          finalTopics = themeGroupsArray.map((group, index) => {
            // Local sentiment analysis for this theme
            let themeSentiments = [];
            
            group.comments.forEach(comment => {
              const score = sentiment(comment.text);
              let classification = 'neutral';
              
              // Business context rules
              const text = comment.text.toLowerCase();
              if (text.includes('expensive') || text.includes('costly') || text.includes('overpriced') || 
                  text.includes('disappointed') || text.includes('terrible') || text.includes('awful') ||
                  text.includes('bad') || text.includes('worst') || text.includes('hate')) {
                classification = 'negative';
              } else if (text.includes('great') || text.includes('excellent') || text.includes('amazing') ||
                         text.includes('love') || text.includes('perfect') || text.includes('wonderful') ||
                         text.includes('best') || text.includes('fantastic')) {
                classification = 'positive';
              } else if (score.comparative > 0.1) {
                classification = 'positive';
              } else if (score.comparative < -0.1) {
                classification = 'negative';
              }
              
              themeSentiments.push({
                classification: classification,
                reasoning: `Fallback analysis: score ${score.comparative}`,
                confidence: 0.7
              });
            });
            
            // Calculate sentiment distribution
            const sentimentCounts = {
              positive: themeSentiments.filter(s => s.classification === 'positive').length,
              negative: themeSentiments.filter(s => s.classification === 'negative').length,
              neutral: themeSentiments.filter(s => s.classification === 'neutral').length
            };
            
            const volume = group.comments.length;
            const percentage = Math.round((volume / comments.length) * 100);
            
            // Overall theme sentiment based on majority
            const sentimentClassification = sentimentCounts.positive > sentimentCounts.negative && sentimentCounts.positive > sentimentCounts.neutral ? 'positive' :
              sentimentCounts.negative > sentimentCounts.positive && sentimentCounts.negative > sentimentCounts.neutral ? 'negative' : 'neutral';
            
            // Calculate average sentiment score for compatibility
            const avgSentiment = sentimentCounts.positive > 0 ? 0.3 : sentimentCounts.negative > 0 ? -0.3 : 0;
            
            return {
              topicId: index + 1,
              title: group.name,
              llmDescription: group.description,
              words: group.keywords.map(keyword => ({ term: keyword, weight: 1, probability: 1 })),
              volume: volume,
              percentage: percentage,
              sentiment: {
                classification: sentimentClassification,
                score: Math.round(avgSentiment * 100) / 100,
                distribution: {
                  positive: sentimentCounts.positive,
                  negative: sentimentCounts.negative,
                  neutral: sentimentCounts.neutral,
                  positivePercentage: Math.round((sentimentCounts.positive / volume) * 100),
                  negativePercentage: Math.round((sentimentCounts.negative / volume) * 100),
                  neutralPercentage: Math.round((sentimentCounts.neutral / volume) * 100)
                }
              },
              confidence: Math.round(group.comments.reduce((sum, c) => sum + c.confidence, 0) / group.comments.length * 100),
              avgWordCount: Math.round(group.comments.reduce((sum, c) => 
                sum + c.text.split(/\s+/).length, 0) / group.comments.length),
              businessImpact: sentimentClassification === 'negative' && percentage > 10 ? 'high' : 
                percentage > 15 ? 'medium' : 'low',
              comments: group.comments, // ALL comments, not samples
              enhancedByAI: false // Fallback mode
            };
          });
          
          console.log(`Fallback theme classification completed with ${finalTopics.length} themes`);
        }
        
        // Keep ALL comments in the response for full display
        const cleanTopics = finalTopics.map(topic => {
          return {
            ...topic,
            sampleQuotes: topic.comments.map(c => c.text) // All comments as "sample quotes"
          };
        });
        
        // Sort by business impact and volume
        cleanTopics.sort((a, b) => {
          const impactOrder = { high: 3, medium: 2, low: 1 };
          const impactDiff = impactOrder[b.businessImpact] - impactOrder[a.businessImpact];
          return impactDiff !== 0 ? impactDiff : b.volume - a.volume;
        });
        
        // Overall sentiment analysis
        const sentimentAnalysis = comments.map(comment => {
          try {
            const score = sentiment(comment);
            return {
              text: comment.substring(0, 100) + '...',
              score: score?.score || 0,
              comparative: Math.round((score?.comparative || 0) * 100) / 100,
              classification: (score?.comparative || 0) > 0.1 ? 'positive' : 
                (score?.comparative || 0) < -0.1 ? 'negative' : 'neutral'
            };
          } catch (error) {
            console.warn('Sentiment analysis failed for comment:', comment?.substring(0, 50));
            return {
              text: comment.substring(0, 100) + '...',
              score: 0,
              comparative: 0,
              classification: 'neutral'
            };
          }
        });
        
        const avgWordCount = Math.round(processedComments.reduce((sum, item) => 
          sum + item.wordCount, 0) / processedComments.length);
        
        fs.unlinkSync(filePath);
        
        console.log(`Analysis complete! Returning ${cleanTopics.length} topics`);
        
        res.json({
          success: true,
          data: {
            totalComments: comments.length,
            coherenceScore: Math.round(coherenceScore * 100) / 100,
            avgWordCount,
            processingTime: Math.round((Date.now() - startTime) / 1000 * 10) / 10,
            topics: cleanTopics,
            metadata: {
              aiEnhanced: cleanTopics.some(t => t.enhancedByAI),
              totalTopics: cleanTopics.length,
              highPriorityCount: cleanTopics.filter(t => t.businessImpact === 'high').length
            },
            sentiment: {
              overall: {
                positive: sentimentAnalysis.filter(s => s.classification === 'positive').length,
                negative: sentimentAnalysis.filter(s => s.classification === 'negative').length,
                neutral: sentimentAnalysis.filter(s => s.classification === 'neutral').length
              },
              details: sentimentAnalysis.slice(0, 20)
            }
          }
        });
        
      } catch (error) {
        console.error('Analysis error:', error);
        fs.unlinkSync(filePath);
        res.status(500).json({ 
          success: false, 
          error: 'Analysis failed: ' + error.message 
        });
      }
    })
    .on('error', (error) => {
      console.error('File processing error:', error);
      fs.unlinkSync(filePath);
      res.status(500).json({ 
        success: false, 
        error: 'File processing failed: ' + error.message 
      });
    });
});

// Real chatbot API endpoint with Claude integration
app.post('/api/chat', express.json(), async (req, res) => {
  try {
    const { message, analysisResults } = req.body;
    
    if (!message) {
      return res.status(400).json({ success: false, error: 'Message is required' });
    }
    
    if (!anthropic) {
      return res.status(500).json({ success: false, error: 'Claude API not configured' });
    }

    // Prepare analysis context for the chatbot
    console.log('Chat API - Received analysisResults:', JSON.stringify(analysisResults, null, 2));
    const analysisContext = analysisResults ? {
      totalComments: analysisResults.totalComments,
      themes: analysisResults.topics?.map(topic => ({
        name: topic.title,
        percentage: topic.percentage,
        volume: topic.volume,
        sentiment: topic.sentiment?.classification,
        description: topic.llmDescription,
        sentimentBreakdown: {
          positive: topic.sentiment?.distribution?.positive || 0,
          negative: topic.sentiment?.distribution?.negative || 0,
          neutral: topic.sentiment?.distribution?.neutral || 0,
          positivePercentage: topic.sentiment?.distribution?.positivePercentage || 0,
          negativePercentage: topic.sentiment?.distribution?.negativePercentage || 0,
          neutralPercentage: topic.sentiment?.distribution?.neutralPercentage || 0
        }
      })) || []
    } : null;

    const chatPrompt = `You are an AI assistant analyzing comment data. You have access to the following analysis results:

${analysisContext ? `
ANALYSIS DATA:
- Total Comments: ${analysisContext.totalComments}
- Themes Identified: ${analysisContext.themes.length}

THEMES BREAKDOWN:
${analysisContext.themes.map(theme => `
• ${theme.name}: ${theme.volume} comments (${theme.percentage}%)
  - Overall Sentiment: ${theme.sentiment}
  - Breakdown: ${theme.sentimentBreakdown.positive} positive (${theme.sentimentBreakdown.positivePercentage}%), ${theme.sentimentBreakdown.negative} negative (${theme.sentimentBreakdown.negativePercentage}%), ${theme.sentimentBreakdown.neutral} neutral (${theme.sentimentBreakdown.neutralPercentage}%)
  - Description: ${theme.description}
`).join('')}
` : 'No analysis data available. Please ask the user to upload and analyze a CSV file first.'}

User Question: ${message}

Instructions:
1. ONLY use the provided analysis data above to answer questions
2. Be specific with numbers and percentages from the data
3. If asked about themes not in the data, say they weren't found
4. Provide actionable insights based on the sentiment and volume data
5. Keep responses concise but informative

Answer the user's question based solely on this analysis data:`;

    const response = await anthropic.messages.create({
      model: "claude-3-haiku-20240307",
      max_tokens: 500,
      temperature: 0.3,
      messages: [{ role: "user", content: chatPrompt }]
    });

    res.json({
      success: true,
      response: response.content[0].text
    });

  } catch (error) {
    console.error('Chat API error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to generate chat response: ' + error.message
    });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development',
    anthropicConfigured: !!anthropic
  });
});



app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});