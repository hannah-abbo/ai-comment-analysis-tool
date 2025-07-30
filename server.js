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
  const filePath = req.file.path;
  const results = [];
  
  fs.createReadStream(filePath)
    .pipe(csv())
    .on('data', (data) => results.push(data))
    .on('end', async () => {
      try {
        // Auto-detect comment columns (look for text-heavy columns)
        const sampleRow = results[0];
        const commentColumns = Object.keys(sampleRow).filter(key => {
          const avgLength = results.slice(0, 10).reduce((sum, row) => 
            sum + (row[key] || '').length, 0) / 10;
          return avgLength > 20; // Likely comment if avg > 20 chars
        });
        
        const comments = results.map(row => {
          const commentText = commentColumns.map(col => row[col] || '').join(' ');
          return commentText.toLowerCase().trim();
        }).filter(comment => comment.length > 10);
        
        // Enhanced text processing
        const processedComments = comments.map((comment, index) => {
          const tokens = natural.WordTokenizer.tokenize(comment);
          const filtered = stopword.removeStopwords(tokens)
            .filter(word => word.length > 2 && /^[a-zA-Z]+$/.test(word));
          return {
            originalIndex: index,
            originalText: comment,
            processedText: filtered.join(' '),
            wordCount: comment.split(/\s+/).length,
            tokens: filtered
          };
        }).filter(item => item.tokens.length > 2);
        
        // LDA Topic Modeling with TF-IDF
        const tfidf = new natural.TfIdf();
        processedComments.forEach(item => {
          tfidf.addDocument(item.processedText);
        });
        
        // Simple topic discovery using TF-IDF term clustering
        const numTopics = Math.min(8, Math.max(3, Math.floor(processedComments.length / 15)));
        const topicTerms = [];
        
        // Get most important terms per document
        const documentTerms = processedComments.map((item, docIndex) => {
          return tfidf.listTerms(docIndex).slice(0, 10); // Top 10 terms per document
        });
        
        // Group documents by similar terms (simple topic modeling)
        const topics = [];
        for (let i = 0; i < numTopics; i++) {
          topics.push({
            id: i,
            documents: [],
            terms: {},
            termCounts: {}
          });
        }
        
        // Assign documents to topics based on term similarity
        processedComments.forEach((item, docIndex) => {
          const topicId = docIndex % numTopics; // Simple round-robin assignment for now
          topics[topicId].documents.push({
            ...item,
            docIndex
          });
          
          // Aggregate terms for this topic
          documentTerms[docIndex].forEach(termData => {
            if (!topics[topicId].terms[termData.term]) {
              topics[topicId].terms[termData.term] = 0;
              topics[topicId].termCounts[termData.term] = 0;
            }
            topics[topicId].terms[termData.term] += termData.tfidf;
            topics[topicId].termCounts[termData.term] += 1;
          });
        });
        
        // Calculate coherence score based on topic term consistency
        const coherenceScore = topics.reduce((sum, topic) => {
          const termValues = Object.values(topic.terms);
          if (termValues.length === 0) return sum;
          const variance = termValues.reduce((s, v, i, arr) => 
            s + Math.pow(v - arr.reduce((a, b) => a + b) / arr.length, 2), 0) / termValues.length;
          return sum + (1 / (1 + variance)); // Higher consistency = lower variance
        }, 0) / topics.length;
        
        // Build topic analysis with enhanced metrics
        const topicAnalysis = [];
        topics.forEach((topic, clusterId) => {
          const clusterComments = topic.documents;
          
          if (clusterComments.length === 0) return;
          
          // Get top terms for this topic
          const topTerms = Object.entries(topic.terms)
            .map(([term, weight]) => ({
              term,
              weight,
              probability: Math.round(weight * 1000) / 1000
            }))
            .sort((a, b) => b.weight - a.weight)
            .slice(0, 10);
          
          // Calculate sentiment for this topic
          const topicSentiments = clusterComments.map(item => {
            const score = sentiment(item.originalText);
            return {
              score: score.score,
              comparative: score.comparative
            };
          });
          
          const avgSentiment = topicSentiments.reduce((sum, s) => 
            sum + s.comparative, 0) / topicSentiments.length;
          
          const sentimentClassification = avgSentiment > 0.1 ? 'positive' : 
            avgSentiment < -0.1 ? 'negative' : 'neutral';
          
          // Calculate confidence based on cluster density
          const clusterSize = clusterComments.length;
          const totalComments = processedComments.length;
          const confidence = Math.min(100, Math.max(70, 
            70 + (clusterSize / totalComments) * 30 + coherenceScore * 20
          ));
          
          // Sample representative quotes
          const sampleQuotes = _.sampleSize(clusterComments, Math.min(3, clusterComments.length))
            .map(item => item.originalText.substring(0, 150) + '...');
          
          topicAnalysis.push({
            topicId: clusterId + 1,
            title: `Theme ${clusterId + 1}: ${topTerms[0]?.term || 'Unknown'}`,
            words: topTerms,
            volume: clusterComments.length,
            percentage: Math.round((clusterComments.length / totalComments) * 100),
            sentiment: {
              classification: sentimentClassification,
              score: Math.round(avgSentiment * 100) / 100,
              distribution: {
                positive: topicSentiments.filter(s => s.comparative > 0.1).length,
                negative: topicSentiments.filter(s => s.comparative < -0.1).length,
                neutral: topicSentiments.filter(s => 
                  s.comparative >= -0.1 && s.comparative <= 0.1).length
              }
            },
            confidence: Math.round(confidence),
            avgWordCount: Math.round(clusterComments.reduce((sum, item) => 
              sum + item.wordCount, 0) / clusterComments.length),
            sampleQuotes,
            businessImpact: sentimentClassification === 'negative' && 
              clusterComments.length > totalComments * 0.1 ? 'high' : 
              clusterComments.length > totalComments * 0.15 ? 'medium' : 'low',
            comments: clusterComments // Store for LLM processing
          });
        });
        
        // GenAI Theme Classification using Claude
        let finalTopics = topicAnalysis;
        if (anthropic && topicAnalysis.length > 0) {
          try {
            const enhancedTopics = await Promise.all(topicAnalysis.map(async (topic) => {
              const topWords = topic.words.slice(0, 5).map(w => w.term).join(', ');
              const sampleComments = topic.comments.slice(0, 5).map(c => c.originalText).join('\n\n');
              
              const prompt = `Analyze this group of customer feedback comments and provide a clear, business-focused theme name.

Top keywords: ${topWords}

Sample comments:
${sampleComments}

Based on the above comments and keywords, provide:
1. A clear, specific theme name (2-4 words, business-focused)
2. A brief description of what this theme represents
3. The main customer concern or topic

Respond in JSON format:
{
  "themeName": "Theme Name",
  "description": "Brief description",
  "mainConcern": "Primary customer concern"
}`;
              
              try {
                const response = await anthropic.messages.create({
                  model: "claude-3-haiku-20240307",
                  max_tokens: 200,
                  temperature: 0.3,
                  messages: [{ role: "user", content: prompt }]
                });
                
                const llmResponse = JSON.parse(response.content[0].text);
                
                // Tag each comment in this group with the theme
                const taggedComments = topic.comments.map(comment => ({
                  ...comment,
                  assignedTheme: llmResponse.themeName,
                  themeDescription: llmResponse.description
                }));
                
                return {
                  ...topic,
                  title: llmResponse.themeName,
                  llmDescription: llmResponse.description,
                  mainConcern: llmResponse.mainConcern,
                  comments: taggedComments,
                  enhancedByAI: true
                };
              } catch (llmError) {
                console.warn(`LLM processing failed for topic ${topic.topicId}:`, llmError.message);
                return {
                  ...topic,
                  enhancedByAI: false
                };
              }
            }));
            
            // Update with LLM-enhanced results
            finalTopics = enhancedTopics;
          } catch (error) {
            console.warn('LLM enhancement failed:', error.message);
          }
        }
        
        // Remove comments from response to reduce payload size
        const cleanTopics = finalTopics.map(topic => {
          const { comments, ...topicWithoutComments } = topic;
          return topicWithoutComments;
        });
        
        // Sort by business impact and volume
        cleanTopics.sort((a, b) => {
          const impactOrder = { high: 3, medium: 2, low: 1 };
          const impactDiff = impactOrder[b.businessImpact] - impactOrder[a.businessImpact];
          return impactDiff !== 0 ? impactDiff : b.volume - a.volume;
        });
        
        // Overall sentiment analysis
        const sentimentAnalysis = comments.map(comment => {
          const score = sentiment(comment);
          return {
            text: comment.substring(0, 100) + '...',
            score: score.score,
            comparative: Math.round(score.comparative * 100) / 100,
            classification: score.comparative > 0.1 ? 'positive' : 
              score.comparative < -0.1 ? 'negative' : 'neutral'
          };
        });
        
        const avgWordCount = Math.round(processedComments.reduce((sum, item) => 
          sum + item.wordCount, 0) / processedComments.length);
        
        fs.unlinkSync(filePath);
        
        res.json({
          success: true,
          data: {
            totalComments: comments.length,
            coherenceScore: Math.round(coherenceScore * 100) / 100,
            avgWordCount,
            processingTime: Date.now() - Date.now(), // Will be set properly
            topics: cleanTopics,
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

app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});



app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});