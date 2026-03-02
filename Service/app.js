const fs = require('fs');
const express = require('express');
const { send } = require('process');

const app = express();

const questions = JSON.parse(fs.readFileSync(`${__dirname}/data/questions.json`, 'utf-8'));

const getAllQuestion = (req, res) => {
    res.status(200).json({
        status: 'success',
        results: questions.length,
        data: questions
    });
}

const getRandomQuestion = (req, res) => {
    const questionsArray = questions.questions;
    const randomIndex = Math.floor(Math.random() * questionsArray.length);
    const randomQuestion = questionsArray[randomIndex];

    res.status(200).json({
        status: 'success',
        data: randomQuestion
    });
}

const sendAnswer = (req, res) => {
    res.status(200)
    .json({status: 'success', data: {score: 8}});
}

app.get('/api/v1/allQuestions', getAllQuestion);
app.get('/api/v1/randomQuestion', getRandomQuestion);
app.post('/api/v1/answer/:questionId', sendAnswer);

const port = 3000;
app.listen(port, () => {
    console.log(`App running on port ${port}...`);
});
