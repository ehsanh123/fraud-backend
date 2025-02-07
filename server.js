const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { execFile } = require('child_process');
const port = process.env.PORT || 5000;


const app = express();
app.use(cors()); // Allow requests from Next.js frontend
app.use(bodyParser.json());

app.post('/api/submit', (req, res) => {
    const formData = req.body;
    const scriptPath = './script.py'; 
    const pythonPath = "python";
    p="prediction : 0.996" 

    // return res.status(200).json({ output: p });  // Respond with the output of the script
    execFile(pythonPath, [scriptPath, JSON.stringify(formData)], (error, stdout, stderr) => {

        if (error) {
            return res.status(500).json({ error: "Failed to execute script", details: error.message });
        }

        if (stderr) {
            return res.status(500).json({ error: "Script error", details: stderr });
        }

        return res.status(200).json(stdout);
    });
});


app.get('/api/submit', (req, res) => {
    const formData = req.body;
    
    p={"prediction" : "0.996" }

    return res.status(200).json({ output: "p" });  // Respond with the output of the script
    // execFile(pythonPath, [scriptPath, JSON.stringify(formData)], (error, stdout, stderr) => {
    //     if (error) {
    //         return res.status(500).json({ error: "Failed to execute script", details: error.message });
    //     }
    //     if (stderr) {
    //         return res.status(500).json({ error: "Script error", details: stderr });
    //     }
    //     return res.json({ output: stdout });
    // });
});

app.get('/', (req, res) => {
    return res.status(200).json({ output: "Hello World"  });  // Respond with the output of the script
    
});


app.listen(port, () => 
    console.log(`Backend running on port ${port}`)
);
